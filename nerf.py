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
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)


class NerfModule(nn.Module):
    def __init__(
            self,
            dim_pts_input=3,
            dim_dir_input=3,
            pts_num_freqs=10,
            dir_num_freqs=4,
            num_layers=8,
            dim_filter=256,
            skip=[4]):
        super().__init__()
        self.skip = skip

        self.pts_encoder = PositionalEncoder(dim_pts_input, pts_num_freqs)
        self.dir_encoder = PositionalEncoder(dim_dir_input, dir_num_freqs)

        self.layers = nn.ModuleList(
            [nn.Linear(self.pts_encoder.dim_output, dim_filter)] + 
            [nn.Linear(dim_filter + self.pts_encoder.dim_output, dim_filter) if i in skip \
             else nn.Linear(dim_filter, dim_filter) for i in range(num_layers - 1)]
        )

        self.sigma = nn.Linear(dim_filter, 1)
        self.rgb_filters = nn.Linear(dim_filter, dim_filter)
        self.branch = nn.Linear(dim_filter + self.dir_encoder.dim_output, dim_filter // 2)
        self.output = nn.Linear(dim_filter // 2, 3)
    
    def forward(self, pts, dirs):
        pts = self.pts_encoder(pts)
        dirs = self.dir_encoder(dirs)

        _x = pts
        for i, layer in enumerate(self.layers):
            _x = torch.relu(layer(_x))
            if i in self.skip:
                _x = torch.cat([pts, _x], dim=-1)
        
        sigma = self.sigma(_x)
        _x = self.rgb_filters(_x)
        _x = torch.cat([_x, dirs], dim=-1)
        _x = torch.relu(self.branch(_x))
        colors = self.output(_x)

        return colors, sigma


def raw2outputs(colors, sigma, z_vals, raw_noise_std):
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat([distances, 1e10 * torch.ones_like(distances[..., :1])], dim=-1)

    sigma = sigma.view(distances.shape)
    colors = colors.view((*distances.shape, 3))

    if raw_noise_std > 0:
        noise = torch.randn(sigma.shape, device=sigma.device) * raw_noise_std
    else:
        noise = 0
    
    alpha = 1 - torch.exp(-nn.functional.relu(sigma + noise) * distances)
    weights = alpha * cumprod_exclusive(1 - alpha + 1e-10)
    rgb = torch.sigmoid(colors)

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    acc_map = torch.sum(weights, dim=-1)
    rgb_map = rgb_map + (1. - acc_map[..., None])
    depth_map = torch.sum(weights * z_vals, dim=-1)

    return rgb_map, depth_map, acc_map, weights


def nerf_forward(
        coarse_model,
        fine_model,
        ray_origins,
        ray_dirs,
        near,
        far,
        num_samples_stratified,
        num_samples_hierarchical,
        perturb,
        raw_noise_std,
        chunk_size=2**18):
    pts, z_vals = sample_stratified(ray_origins, ray_dirs, near, far, num_samples_stratified, perturb)
    dirs = ray_dirs[:, None, ...].expand(pts.shape).reshape((-1, 3))
    pts = pts.reshape((-1, 3))

    pts_chunks = chunkify(pts, chunk_size)
    dir_chunks = chunkify(dirs, chunk_size)
    color_chunks = []
    sigma_chunks = []
    for pts_chunk, dir_chunks in zip(pts_chunks, dir_chunks):
        colors, sigma = coarse_model(pts_chunk, dir_chunks)
        color_chunks.append(colors)
        sigma_chunks.append(sigma)
    colors = torch.cat(color_chunks)
    sigma = torch.cat(sigma_chunks)

    rgb_map_coarse, _, _, weights_coarse = raw2outputs(colors, sigma, z_vals, raw_noise_std)

    pts, z_vals_combined, _ = sample_hierarchical(ray_origins, ray_dirs, z_vals, weights_coarse, num_samples_hierarchical, perturb)
    dirs = ray_dirs[:, None, ...].expand(pts.shape).reshape((-1, 3))
    pts = pts.reshape((-1, 3))

    pts_chunks = chunkify(pts, chunk_size)
    dir_chunks = chunkify(dirs, chunk_size)
    color_chunks = []
    sigma_chunks = []
    for pts_chunk, dir_chunks in zip(pts_chunks, dir_chunks):
        colors, sigma = fine_model(pts_chunk, dir_chunks)
        color_chunks.append(colors)
        sigma_chunks.append(sigma)
    colors = torch.cat(color_chunks)
    sigma = torch.cat(sigma_chunks)

    rgb_map_fine, _, _, _ = raw2outputs(colors, sigma, z_vals_combined, raw_noise_std)

    return rgb_map_coarse, rgb_map_fine


def train(
        dataloader,
        coarse_model,
        fine_model,
        near=2.0,
        far=6.0,
        num_samples_stratified=64,
        num_samples_hierarchical=128,
        batch_size=2**12):
    device = next(coarse_model.parameters()).device

    optimizer = torch.optim.AdamW(
        list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=2e-4,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/100_000))
    scaler = torch.amp.GradScaler("cuda")

    itr = 22001
    while True:
        rgb, ray_origins, ray_dirs = dataloader.sample(batch_size)

        rgb = rgb.to(device)
        ray_origins = ray_origins.to(device)
        ray_dirs = ray_dirs.to(device)

        with torch.amp.autocast("cuda"): 
            rgb_coarse, rgb_fine = nerf_forward(
                coarse_model,
                fine_model,
                ray_origins,
                ray_dirs,
                near,
                far,
                num_samples_stratified,
                num_samples_hierarchical,
                True,
                1
            )

        coarse_loss = nn.functional.mse_loss(rgb, rgb_coarse)
        fine_loss = nn.functional.mse_loss(rgb, rgb_fine)
        loss = coarse_loss + fine_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()

        coarse_psnr = -10.0 * torch.log(coarse_loss) / np.log(10.0)
        fine_psnr = -10.0 * torch.log(fine_loss) / np.log(10.0)
        print(f"| step={itr} | c_loss={coarse_loss:.5f} | c_psnr={coarse_psnr:.2f} | f_loss={fine_loss:.5f} | f_psnr={fine_psnr:.2f} |")

        if itr % 2000 == 0:
            i = np.random.randint(0, len(dataloader))
            K, c2w, ray_origins, ray_dirs, img = dataloader.load_image(i)

            rgb_coarse, rgb_fine = render_image(
                coarse_model,
                fine_model,
                ray_origins,
                ray_dirs,
                near,
                far,
                num_samples_stratified,
                num_samples_hierarchical
            )

            img = img.numpy()
            rgb_coarse = rgb_coarse.numpy()
            rgb_fine = rgb_fine.numpy()

            img = (cv2.cvtColor(np.concat([img, rgb_coarse, rgb_fine], axis=1), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
            cv2.imwrite(f"./outputs/{itr}_{i}.png", img)
            
        
        if itr % 2000 == 0:
            torch.save(coarse_model.state_dict(), f"./models/lego_coarse_{itr}.pth")
            torch.save(fine_model.state_dict(), f"./models/lego_fine_{itr}.pth")

        itr += 1

def render_image(
        coarse_model,
        fine_model,
        ray_origins,
        ray_dirs,
        near,
        far,
        num_samples_stratified,
        num_samples_hierarchical,
        chunk_size=2**11):
    dims = ray_origins.shape[:2]
    ray_origins = ray_origins.reshape(-1, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            ray_origins_chunks = chunkify(ray_origins, chunk_size)
            ray_dirs_chunks = chunkify(ray_dirs, chunk_size)

            coarse_chunks = []
            fine_chunks = []
            for (rays_o, rays_d) in zip(ray_origins_chunks, ray_dirs_chunks):
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)

                rgb_coarse, rgb_fine = nerf_forward(
                    coarse_model,
                    fine_model,
                    rays_o,
                    rays_d,
                    near,
                    far,
                    num_samples_stratified,
                    num_samples_hierarchical,
                    False,
                    0
                )

                coarse_chunks.append(rgb_coarse.cpu())
                fine_chunks.append(rgb_fine.cpu())
            
            rgb_coarse = torch.cat(coarse_chunks).reshape(*dims, 3)
            rgb_fine = torch.cat(fine_chunks).reshape(*dims, 3)
    
    return rgb_coarse, rgb_fine
            


if __name__ == "__main__":
    device = "cuda"

    dataset = NeRFDataset(os.path.join("./dataset", "nerf_synthetic", "lego"))

    dataloader = NeRFRayDataLoader(dataset, "train")


    coarse_model = NerfModule()
    fine_model = NerfModule()

    coarse_model.load_state_dict(torch.load(f"./models/lego_coarse_22000.pth"))
    fine_model.load_state_dict(torch.load(f"./models/lego_fine_22000.pth"))

    coarse_model = coarse_model.to(device)
    fine_model = fine_model.to(device)

    near = 2 / dataloader.dataset.data[dataloader.split]["bound_norm_scale"]
    far = 6 / dataloader.dataset.data[dataloader.split]["bound_norm_scale"]
    

    train(dataloader, coarse_model, fine_model, near, far)
    # show(dataloader, coarse_model, fine_model, 2.0, 6.0, 64, 128)

