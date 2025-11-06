import torch
from torch import nn
from dataset import *
from helpers import *


class PositionalEncoder(nn.Module):
    def __init__(self, dim_input, num_freqs):
        super().__init__()
        self.dim_output = dim_input * (1 + 2 * num_freqs)
        self.embed_fns = [lambda x: x]

        freq_bands = 2 ** torch.linspace(0, num_freqs -1, num_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
    
    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class NerfModule(nn.Module):
    def __init__(self, dim_input=3, dim_view_dirs=3, num_layers=8, dim_filter=256, skip=[4]):
        super().__init__()
        self.skip = skip

        self.layers = nn.ModuleList(
            [nn.Linear(dim_input, dim_filter)] + 
            [nn.Linear(dim_filter + dim_input, dim_filter) if i in skip \
             else nn.Linear(dim_filter, dim_filter) for i in range(num_layers - 1)]
        )

        self.alpha = nn.Linear(dim_filter, 1)
        self.rgb_filters = nn.Linear(dim_filter, dim_filter)
        self.branch = nn.Linear(dim_filter + dim_view_dirs, dim_filter // 2)
        self.output = nn.Linear(dim_filter // 2, 3)
    
    def forward(self, pts, view_dirs):
        _x = pts
        for i, layer in enumerate(self.layers):
            _x = torch.relu(layer(_x))
            if i in self.skip:
                _x = torch.cat([pts, _x], dim=-1)
        
        alpha = self.alpha(_x)
        _x = self.rgb_filters(_x)
        _x = torch.cat([_x, view_dirs], dim=-1)
        _x = torch.relu(self.branch(_x))
        rgb = self.output(_x)

        out = torch.cat([rgb, alpha], dim=-1)
        return out


def raw2outputs(raw, z_vals, raw_noise_std):
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat([distances, 1e10 * torch.ones_like(distances[..., :1])], dim=-1)

    if raw_noise_std > 0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
    else:
        noise = 0
    
    alpha = 1 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * distances)
    weights = alpha * cumprod_exclusive(1 - alpha + 1e-10)
    rgb = torch.sigmoid(raw[..., :3])

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    # disp_map = 1 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, dim=-1))
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map, weights


def chunkify(x, chunk_size):
    return [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]


def prepare_pts_chunks(pts, encoding_fn, chunk_size):
    pts = pts.reshape((-1, 3))
    pts = encoding_fn(pts)
    pts = chunkify(pts, chunk_size)
    return pts


def prepare_view_dirs_chunks(pts, ray_dirs, encoding_fn, chunk_size):
    view_dirs = ray_dirs[:, None, ...].expand(pts.shape).reshape((-1, 3))
    view_dirs = encoding_fn(view_dirs)
    view_dirs = chunkify(view_dirs, chunk_size)
    return view_dirs


def nerf_forward(
        ray_origins,
        ray_dirs,
        near,
        far,
        num_samples_stratified,
        num_samples_hierarchical,
        coarse_model,
        fine_model,
        pts_encoding_fn,
        view_dirs_encoding_fn,
        perturb,
        raw_noise_std,
        chunk_size=2**15):
    pts, z_vals = sample_stratified(ray_origins, ray_dirs, near, far, num_samples_stratified, perturb)
    pts_batches = prepare_pts_chunks(pts, pts_encoding_fn, chunk_size)
    view_dir_batches = prepare_view_dirs_chunks(pts, ray_dirs, view_dirs_encoding_fn, chunk_size)

    raw = []
    for pts_batch, view_dir_batch in zip(pts_batches, view_dir_batches):
        raw.append(coarse_model(pts_batch, view_dir_batch))
    raw = torch.cat(raw, dim=0)
    raw = raw.reshape(list(pts.shape[:2]) + [raw.shape[-1]])

    rgb_map0, _, _, weights = raw2outputs(raw, z_vals, raw_noise_std)

    if num_samples_hierarchical > 0:

        pts, z_vals_combined, _ = sample_hierarchical(ray_origins, ray_dirs, z_vals, weights, num_samples_hierarchical, perturb)

        pts_batches = prepare_pts_chunks(pts, pts_encoding_fn, chunk_size)
        view_dir_batches = prepare_view_dirs_chunks(pts, ray_dirs, view_dirs_encoding_fn, chunk_size)

        raw = []
        for pts_batch, view_dir_batch in zip(pts_batches, view_dir_batches):
            raw.append(fine_model(pts_batch, view_dir_batch))
        raw = torch.cat(raw, dim=0)
        raw = raw.reshape(list(pts.shape[:2]) + [raw.shape[-1]])

        rgb_map, _, _, _ = raw2outputs(raw, z_vals_combined, raw_noise_std)

        return {"rgb_map0": rgb_map0, "rgb_map": rgb_map}
    return {"rgb_map0": rgb_map0}


def train(dataloader, coarse_model, fine_model, pts_encoder, view_encoder, near=2, far=6, batch_size=4096):
    model_device = next(coarse_model.parameters()).device

    optimizer = torch.optim.AdamW(
        list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=5e-4,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/100_000))
    scaler = torch.amp.GradScaler("cuda")

    itr = 0
    while True:
        pixels, ray_origins, ray_dirs = dataloader.sample(batch_size)

        pixels = pixels.to(model_device)
        ray_origins = ray_origins.to(model_device)
        ray_dirs = ray_dirs.to(model_device)

        with torch.amp.autocast("cuda"): 
            outputs = nerf_forward(
                ray_origins,
                ray_dirs,
                near,
                far,
                64,
                128,
                coarse_model,
                fine_model,
                pts_encoder,
                view_encoder,
                True,
                1
            )

            loss =  nn.functional.mse_loss(outputs["rgb_map0"], pixels) + \
                    nn.functional.mse_loss(outputs["rgb_map"], pixels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()

        print(itr, loss.item())

        if itr % 2000 == 0:
            i = np.random.randint(0, len(dataloader))
            K, c2w, ray_origins, ray_dirs, img = dataloader.load_image(i, 0.8)

            ray_origins = ray_origins.reshape((-1, 3))
            ray_dirs = ray_dirs.reshape((-1, 3))

            ray_origins_chunks = chunkify(ray_origins, 2**15)
            ray_dirs_chunks = chunkify(ray_dirs, 2**15)

            rgb0_chunks = []
            rgb_chunks = []
            with torch.no_grad():
                for ray_origins, ray_dirs in zip(ray_origins_chunks, ray_dirs_chunks):
                    ray_origins = ray_origins.to(model_device)
                    ray_dirs = ray_dirs.to(model_device)
                    outputs = nerf_forward(
                        ray_origins,
                        ray_dirs,
                        near,
                        far,
                        64,
                        128,
                        coarse_model,
                        fine_model,
                        pts_encoder,
                        view_encoder,
                        False,
                        0,
                        chunk_size=2**15
                    )
                    rgb0_chunks.append(outputs["rgb_map0"].cpu())
                    rgb_chunks.append(outputs["rgb_map"].cpu())

            coarse_img = torch.cat(rgb0_chunks, dim=0).reshape(img.shape)
            fine_img = torch.cat(rgb_chunks, dim=0).reshape(img.shape)

            img = (cv2.cvtColor(np.concat([img, coarse_img, fine_img], axis=1), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
            cv2.imwrite(f"./outputs/{itr}_{i}.png", img)    
        
        if itr % 4000 == 0:
            torch.save(coarse_model.state_dict(), f"./models/lego_coarse_{itr}.pth")
            torch.save(fine_model.state_dict(), f"./models/lego_fine_{itr}.pth")

        cv2.waitKey(1)
        itr += 1

if __name__ == "__main__":
    device = "cuda"

    dataset = NeRFDataset(os.path.join("./dataset", "nerf_synthetic", "lego"))

    dataloader = NeRFRayDataLoader(dataset, "train")

    pts_encoder = PositionalEncoder(3, 10)
    view_encoder = PositionalEncoder(3, 4)

    coarse_model = NerfModule(pts_encoder.dim_output, view_encoder.dim_output, 2).to(device)
    fine_model = NerfModule(pts_encoder.dim_output, view_encoder.dim_output, 6).to(device)

    train(dataloader, coarse_model, fine_model, pts_encoder, view_encoder)
