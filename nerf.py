import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from dataset import *
from helpers import *
from torchsummary import summary

class Embedder:
    def __init__(self, multires, input_dims=3):
        self.embed_fns = []
        self.out_dim = 0

        self.embed_fns.append(lambda x: x)
        self.out_dim += input_dims

        num_freq = multires
        max_freq_log2 = multires - 1
        freq_bands = 2 ** torch.linspace(0, max_freq_log2, num_freq)
        for freq in freq_bands:
            for fn in [torch.sin, torch.cos]:
                self.embed_fns.append(lambda x, pfn=fn, freq=freq: pfn(x * freq))
                self.out_dim += input_dims
    
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


class NeRF_Model(nn.Module):
    def __init__(
            self,
            layer_depth=8,
            layer_width=128,
            num_input_channels=3,
            num_input_view_channels=3,
            pts_embed_multires=10,
            view_embed_multires=4,
            skips=[4]):
        super().__init__()

        self.pts_embedder = Embedder(pts_embed_multires, num_input_channels)
        self.view_embbedder = Embedder(view_embed_multires, num_input_view_channels)
        
        self.num_input_channels = num_input_channels
        self.num_input_view_channels = num_input_view_channels
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pts_embedder.out_dim, layer_width)] + 
            [nn.Linear(
                layer_width if i not in skips else layer_width + self.pts_embedder.out_dim,
                layer_width) for i in range(layer_depth -1)])
    
        self.views_linears = nn.ModuleList([nn.Linear(self.view_embbedder.out_dim + layer_width, layer_width // 2)])

        self.feature_linear = nn.Linear(layer_width, layer_width)
        self.alpha_linear = nn.Linear(layer_width, 1)
        self.rgb_linear = nn.Linear(layer_width // 2, 3)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.num_input_channels, self.num_input_view_channels], dim=-1)

        input_pts = self.pts_embedder.embed(input_pts)
        input_views = self.view_embbedder.embed(input_views)

        _x = input_pts
        for i, layer in enumerate(self.pts_linears):
            _x = F.relu(layer(_x))
            if i in self.skips:
                _x = torch.cat([input_pts, _x], dim=-1)
        
        alpha = self.alpha_linear(_x)
        feature = self.feature_linear(_x)
        _x = torch.cat([feature, input_views], dim=-1)

        for layer in self.views_linears:
            _x = F.relu(layer(_x))
        
        rgb = self.rgb_linear(_x)
        outputs = torch.cat([rgb, alpha], dim=-1)
        return outputs



def raw_to_outputs(raw, z_vals, ray_dirs, raw_noise_std):
    distances = z_vals[..., 1:] - z_vals[..., :-1]
    distances = torch.cat([distances, torch.Tensor([1e10]).expand(distances[..., :1].shape).to(distances.device)], dim=-1)
    distances = (distances * torch.norm(ray_dirs[..., None, :], dim=-1)).to(raw.device)

    rgb = torch.sigmoid(raw[..., :3]).reshape((*distances.shape, 3))
    if raw_noise_std > 0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
    else:
        noise = 0

    alpha = 1 - torch.exp(-F.relu(raw[..., 3] + noise).reshape(distances.shape) * distances)
    weights = (alpha * torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1), device=alpha.device),
            1 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1])

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    return rgb_map, weights



def render_ray_batch(models, model_device, ray_origins, ray_dirs, near, far, perturb, num_samples, raw_noise_std, N_importance):
    near = near * torch.ones_like(ray_dirs[..., :1])
    far = far * torch.ones_like(ray_dirs[..., :1])

    t_vals = torch.linspace(0, 1, num_samples)
    z_vals = 1 / ((1 / near) * (1 - t_vals) + (1 / far) * t_vals)

    ray_origins = ray_origins.to(model_device)
    ray_dirs = ray_dirs.to(model_device)

    if perturb:
        midpoints = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([midpoints, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], midpoints], dim=-1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = (lower + (upper - lower) * t_rand)
    
    z_vals = z_vals.to(model_device)
    pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]
    view_dirs1 = ray_dirs.unsqueeze(1).expand(-1, num_samples, -1)

    inputs = torch.cat([pts, view_dirs1], dim=-1).reshape(-1, 6)
    raw = models[0](inputs)
    rgb0, weights0 = raw_to_outputs(raw, z_vals, ray_dirs, raw_noise_std)
    if N_importance > 0:
        z_vals_mid = (0.5 * (z_vals[..., 1:] + z_vals[..., :-1])).to(weights0.device)
        z_samples = sample_pdf(z_vals_mid, weights0[..., 1:-1], N_importance, det=not perturb)
        z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals.to(z_samples.device), z_samples], dim=-1), dim=-1)
        pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]
        view_dirs2 = ray_dirs.unsqueeze(1).expand(-1, z_vals.shape[-1], -1).to(model_device)
        inputs = torch.cat([pts, view_dirs2], dim=-1).reshape(-1, 6)
        raw = models[1](inputs)
        rgb1, _ = raw_to_outputs(raw, z_vals, ray_dirs, raw_noise_std)
        return rgb0, rgb1
    return rgb0



def render(models, model_device, dims, ray_origins, ray_dirs, near=1.0, far=6.0, chunk_size=1024 * 8, num_samples=32, raw_noise_std=0, perturb=False, N_importance=0):
    B = ray_origins.shape[0]
    flat_ro = ray_origins.reshape(-1, 3)
    flat_rd = ray_dirs.reshape(-1, 3)

    N = len(flat_ro)
    mini_batches = int(np.ceil(N / chunk_size))

    rgb0 = []
    rgb1 = []

    for i in range(mini_batches):
        ret = render_ray_batch(
            models,
            model_device,
            flat_ro[i * chunk_size:min((i + 1) * chunk_size, N)],
            flat_rd[i * chunk_size:min((i + 1) * chunk_size, N)],
            near,
            far,
            perturb,
            num_samples,
            raw_noise_std,
            N_importance)
        
        if N_importance > 0:
            im0, im1 = ret
            im0 = im0.to("cpu")
            im1 = im1.to("cpu")
            rgb0.append(im0)
            rgb1.append(im1)
        else:
            im0 = ret
            im0 = im0.to("cpu")
            rgb0.append(im0)
        
    rgb0 = torch.cat(rgb0, dim=0).reshape(B, *dims, 3)
    if N_importance > 0:
        rgb1 = torch.cat(rgb1, dim=0).reshape(B, *dims, 3)
        return rgb0, rgb1
    return rgb0

import cv2

def train(models, dataloader: DataLoader):
    model_device = next(models[0].parameters()).device
    optimizer = torch.optim.AdamW(
        list(models[0].parameters()) + list(models[1].parameters()),
        betas=(0.9, 0.98),
        lr=5e-4,
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1**(1/100_000))
    # scaler = torch.amp.GradScaler("cuda")

    itr = 0
    while True:
        def train_step():
            pixels, ray_origins, ray_dirs = dataloader.sample(2048)
            # with torch.amp.autocast("cuda"):
            rgb0, rgb1 = render_ray_batch(
                models,
                model_device,
                ray_origins,
                ray_dirs,
                1.0,
                6.0,
                True,
                64,
                1,
                128)
            pixels = pixels.to(rgb0.device)
            loss = F.mse_loss(rgb0, pixels) + F.mse_loss(rgb1, pixels)
            
            print(itr, loss.item())
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_step()

        if itr % 2000 == 0:
            torch.save(models[0].state_dict(), f"./models/lego_0_{itr}.pth")
            torch.save(models[1].state_dict(), f"./models/lego_1_{itr}.pth")

        if itr % 1000 == 0:
            path = os.path.join("./outputs")

            with torch.no_grad():
                for i in np.random.randint(0, len(dataloader), 3):
                    K, c2w, ray_origins, ray_dirs, img = dataloader.load_image(i, 0.3)
                    ray_origins = ray_origins[None, ...]
                    ray_dirs = ray_dirs[None, ...]
                    rgb0, rgb1 = render(models, model_device, img.shape[:2], ray_origins, ray_dirs, 1.0, 6.0, num_samples=64, N_importance=128)

                    y_true = img.detach().cpu().numpy()
                    y_pred0 = rgb0[0].detach().cpu().numpy()
                    y_pred1 = rgb1[0].detach().cpu().numpy()

                    img = np.concat([y_true, y_pred0, y_pred1], axis=1)
                    img = (cv2.cvtColor(img, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(path, f"{itr}_{i}.png"), img)    

        itr += 1


        # for K, c2w, ray_origins, ray_dirs, img in dataloader:
        #     print(itr)
        #     dims = img.shape[1:3]

        #     with torch.amp.autocast("cuda"):
        #         step = 1
        #         for i in range(0, dataloader.batch_size, step):
        #             rgb0, rgb1 = render(models, model_device, dims, ray_origins[i:i+step], ray_dirs[i:i+step], train=True, perturb=True, raw_noise_std=1, N_importance=16)
        #             imgd = img[i:i+step].to(model_device)
        #             loss = F.mse_loss(rgb0, imgd) + F.mse_loss(rgb1, imgd)
        #             scaler.scale(loss).backward()

        #         scaler.step(optimizer)
        #         scaler.update()
        #         optimizer.zero_grad()

        #     y_true = img[0].detach().cpu().numpy()
        #     y_pred0 = rgb0[0].detach().cpu().numpy()
        #     y_pred1 = rgb1[0].detach().cpu().numpy()

        #     cv2.imshow("img", np.concat([y_true, y_pred0, y_pred1], axis=0))
        #     cv2.waitKey(1)

            # if itr % 2000 == 0:
            #     torch.save(models[0].state_dict(), f"./drums{itr}.pth")

            # itr += 1

            


if __name__ == "__main__":
    device = torch.device("cuda")

    model_coarse = NeRF_Model().to(device)
    model_fine = NeRF_Model().to(device)

    # width = 500
    # height = 500
    # focal_length = 100

    # K = torch.tensor([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]])
    # c2w = torch.eye(4)
    # c2w[2, 3] = -3.0

    # rgb, depth = render(model, K, c2w)

    # rgb = rgb.numpy()
    
    # import cv2
    # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # cv2.imshow("img", bgr)
    # cv2.waitKey(0)

    dataloader = NeRFRayDataLoader(
        NeRFDataset(os.path.join("./dataset", "nerf_synthetic", "lego")),
        "train"
    )

    # model_coarse.load_state_dict(torch.load(f"./models/lego_0_254000.pth"))
    # model_coarse = model_coarse.to(device)
    # model_fine.load_state_dict(torch.load(f"./models/lego_1_254000.pth"))
    # model_fine = model_fine.to(device)

    train((model_coarse, model_fine), dataloader)

    # summary(model_fine, (6,), batch_size=4096)
    # exit()


    for i in range(100):
        K, c2w, ray_origins, ray_dirs, img = dataloader.load_image(i, 0.4)

        ray_origins = ray_origins[None, ...]
        ray_dirs = ray_dirs[None, ...]

        with torch.no_grad():
            rgb0, rgb1 = render((model_coarse, model_fine), device, img.shape[:2], ray_origins, ray_dirs, num_samples=64, N_importance=256, perturb=False)

        y_true = img.detach().cpu().numpy()
        y_pred0 = rgb0[0].detach().cpu().numpy()
        y_pred1 = rgb1[0].detach().cpu().numpy()
        
        cv2.imshow("img", np.concat([y_true, y_pred0, y_pred1], axis=1))
        cv2.waitKey(1)
        