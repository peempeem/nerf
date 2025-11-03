import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF_Model(nn.Module):
    def __init__(
            self,
            layer_depth=8,
            layer_width=256,
            num_input_channel=3,
            num_input_channel_views=2):
        super().__init__()

        self.num_input_channels = num_input_channel
        self.num_input_channels_views = num_input_channel_views

        self.layers = nn.ModuleList(
            [nn.Linear(num_input_channel, layer_width)] +
            [nn.Linear(layer_width, layer_width) for _ in range(layer_depth - 1)]
        )

        self.view_layers = nn.ModuleList(
            [nn.Linear(num_input_channel_views + layer_width, layer_width // 2)]
        )

        self.feature_layer = nn.Linear(layer_width, layer_width)
        self.density_layer = nn.Linear(layer_width, 1)
        self.rgb_layer = nn.Linear(layer_width // 2, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x,
            [self.num_input_channels, self.num_input_channels_views],
            dim=-1
        )

        _x = input_pts
        for layer in self.layers:
            _x = F.relu(layer(_x))
        
        density = F.softplus(self.density_layer(_x))
        features = self.feature_layer(_x)
        _x = torch.cat([features, input_views], dim=-1)

        for layer in self.view_layers:
            _x = F.relu(layer(_x))

        rgb = self.rgb_layer(_x)
        outputs = torch.cat([rgb, density], dim=-1)
        return outputs


def get_rays(K, c2w):
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)
    i, j = torch.meshgrid(
        torch.linspace(0, height - 1, height),
        torch.linspace(0, width -1, width),
        indexing="ij"
    )
    
    dirs = torch.stack([
        (i - K[0][2]) / K[0][0],
        -(j - K[1][2]) / K[1][1],
        -torch.ones_like(i)], -1)
    
    ray_dirs = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    ray_dirs = F.normalize(ray_dirs, dim=-1)
    ray_origins = c2w[:3, 3].expand(ray_dirs.shape)
    return ray_origins, ray_dirs


def sample_points(ray_origins, ray_dirs, near, far, num_samples):
    z_uniform = torch.linspace(near, far, num_samples, device=ray_origins.device)
    midpoints = 0.5 * (z_uniform[1:] + z_uniform[:-1])
    upper_bounds = torch.cat([midpoints, z_uniform[-1:]], dim=-1)
    lower_bounds = torch.cat([z_uniform[:1], midpoints], dim=-1)
    z_rand = torch.rand(ray_origins.shape[:-1] + (num_samples,), device=ray_origins.device)
    z_strata = lower_bounds + (upper_bounds - lower_bounds) * z_rand
    pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_strata[..., :, None]
    return pts, z_strata


def raw_to_outputs(raw, z_vals, ray_dirs, raw_noise_std=0):
    N_rays = len(raw)
    device = raw.device

    rgb = torch.sigmoid(raw[..., :3])
    sigma = F.softplus(raw[..., 3])

    z_vals = z_vals.to(device)

    if raw_noise_std > 0:
        sigma += torch.randn(sigma.shape, device=device) * raw_noise_std
    
    distances = (z_vals[:, 1:] - z_vals[:, :-1])
    distances = torch.cat([distances, torch.full((N_rays, 1), 1e10, device=device)], dim=-1)
    distances = distances * torch.norm(ray_dirs[:, None, :], dim=-1).to(device)

    alpha = 1 - torch.exp(-sigma * distances)
    transmittance = torch.cat([
        torch.ones(N_rays, 1, device=device),
        1 - alpha[:, :-1] + 1e-10], dim=-1)
    weights = alpha * torch.cumprod(transmittance, dim=-1)

    rgb_map = torch.sum(weights[:, :, None] * rgb, dim=1)
    depth_map = torch.sum(weights * z_vals, dim=-1)

    return rgb_map, depth_map


def render_ray_batch(model, ray_origins, ray_dirs, near, far, num_samples=64, raw_noise_std=0):
    model_device = next(model.parameters()).device

    pts, z_vals = sample_points(ray_origins, ray_dirs, near, far, num_samples)

    flattened_pts = pts.reshape(-1, 3)
    flattened_dirs = ray_dirs.reshape(-1, 3)
    xy_norm = torch.norm(flattened_dirs[:, :2], dim=-1, keepdim=True)
    theta = torch.atan2(flattened_dirs[:, 2], xy_norm.squeeze(-1))
    phi = torch.atan2(flattened_dirs[:, 1], flattened_dirs[:, 0])
    flattened_views = torch.stack([theta, phi], dim=-1)
    flattened_views = flattened_views.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, 2)

    rays = torch.cat([flattened_pts, flattened_views], dim=-1).to(model_device)
    raw = model(rays).reshape(-1, num_samples, 4)
    return raw_to_outputs(raw, z_vals, ray_dirs, raw_noise_std)


def render(model, K, c2w, near=1.0, far=3.0, chunk_size=1024 * 32):
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)
    ray_origins, ray_dirs = get_rays(K, c2w)
    flat_ro = ray_origins.reshape(-1, 3)
    flat_rd = ray_dirs.reshape(-1, 3)

    N = len(flat_ro)
    batches = int(np.ceil(N / chunk_size))

    rgb = []
    depth = []
    with torch.no_grad():
        for i in range(batches):
            rgb_map, depth_map = render_ray_batch(
                model,
                flat_ro[i * chunk_size:min((i + 1) * chunk_size, N)],
                flat_rd[i * chunk_size:min((i + 1) * chunk_size, N)],
                near,
                far)
            rgb_map = rgb_map.to("cpu")
            depth_map = depth_map.to("cpu")
            print(rgb_map.shape, depth_map.shape)
            rgb.append(rgb_map)
            depth.append(depth_map)
    rgb = torch.cat(rgb, dim=0).reshape(height, width, 3)
    depth = torch.cat(depth, dim=0).reshape(height, width)
    return rgb, depth


# def train():



if __name__ == "__main__":
    device = torch.device("cuda")

    model = NeRF_Model().to(device)

    width = 300
    height = 300
    focal_length = 100

    K = torch.tensor([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]])
    c2w = torch.eye(4)
    c2w[2, 3] = -3.0

    rgb, depth = render(model, K, c2w)

    rgb = rgb.numpy()
    
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", bgr)
    cv2.waitKey(0)
