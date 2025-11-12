import numpy as np
import torch


def chunkify(x, chunk_size):
    return [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]


def get_rays(height, width, K, c2w):
    i, j = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -np.ones_like(i)], axis=-1, dtype=np.float32)
    ray_dirs = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    ray_dirs = (ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True))
    ray_origins = np.broadcast_to(c2w[:3, -1], np.shape(ray_dirs))
    return ray_origins, ray_dirs


def sample_stratified(ray_origins, ray_dirs, near, far, n_samples, perturb, lindisp):
    t_vals = torch.linspace(0, 1, n_samples, device=ray_origins.device)
    if lindisp:
        z_vals = 1 / ((1 / near) * (1 - t_vals) + (1 / far) * t_vals)
    else:
        z_vals = near * (1 - t_vals) + far * t_vals

    if perturb:
        midpoints = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([midpoints, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], midpoints], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand
    z_vals = z_vals.expand(list(ray_origins.shape[:-1]) + [n_samples])

    pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def sample_pdf(bins, weights, num_samples, perturb):
    pdf = (weights + 1e-6) / torch.sum(weights + 1e-6, dim=-1, keepdim=True)

    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0, 1, num_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)

    below = torch.clamp(indices - 1, min=0)
    above = torch.clamp(indices, max=cdf.shape[-1] - 1)
    good_indices = torch.stack([below, above], dim=-1)

    matched_shape = list(good_indices.shape[:-1]) + [cdf.shape[-1]]
    good_cdf = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=good_indices)
    good_bins = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=good_indices)

    denominator = (good_cdf[..., 1] - good_cdf[..., 0])
    denominator = torch.where(denominator < 1e-6, torch.ones_like(denominator), denominator)
    t = (u - good_cdf[..., 0]) / denominator
    samples = good_bins[..., 0] + t * (good_bins[..., 1] - good_bins[..., 0])
    
    return samples


def sample_hierarchical(ray_origins, ray_dirs, z_vals, weights, num_samples, perturb):
    z_midpoints = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(z_midpoints, weights[..., 1:-1], num_samples, perturb).detach()

    z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    pts = ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals_combined[..., :, None]
    return pts, z_vals_combined, z_samples


def cumprod_exclusive(x):
    cumprod = torch.cumprod(x, dim=-1)
    cumprod = torch.roll(cumprod, 1, dims=-1)
    cumprod[..., 0] = 1
    return cumprod


def ndc_rays(
        height: float,
        width: float,
        focal: float,
        near: float,
        ray_origins: np.ndarray,
        ray_dirs: np.ndarray):
    t = -(near + ray_origins[..., 2]) / ray_dirs[..., 2]
    ray_origins = ray_origins + t[..., None] * ray_dirs
    
    o0 = -1 / (width / (2 * focal)) * ray_origins[..., 0] / ray_origins[..., 2]
    o1 = -1 / (height / (2 * focal)) * ray_origins[..., 1] / ray_origins[..., 2]
    o2 = 1 + 2 * near / ray_origins[..., 2]

    d0 = -1 / (width / (2 * focal)) * (ray_dirs[..., 0] / ray_dirs[..., 2] - ray_origins[..., 0] / ray_origins[..., 2])
    d1 = -1 / (height / (2 * focal)) * (ray_dirs[..., 1] / ray_dirs[..., 2] - ray_origins[..., 1] / ray_origins[..., 2])
    d2 = -2 * near / ray_origins[..., 2]
    
    ray_origins = np.stack([o0, o1, o2], axis=-1)
    ray_dirs = np.stack([d0,d1,d2], axis=-1)
    
    return ray_origins, ray_dirs