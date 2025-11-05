import numpy as np
import torch


def get_rays(height, width, K, c2w):
    i, j = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    dirs = np.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -np.ones_like(i)], axis=-1)
    ray_dirs = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)
    ray_dirs = (ray_dirs / np.linalg.norm(ray_dirs, axis=-1, keepdims=True))
    ray_origins = np.broadcast_to(c2w[:3, -1], np.shape(ray_dirs))
    return ray_origins, ray_dirs


def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-6
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0, 1, N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    
    u = u.contiguous().to(cdf.device)

    indicies = torch.searchsorted(cdf, u, right=True)
    lower = torch.max(torch.zeros_like(indicies), indicies - 1)
    upper = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indicies), indicies)
    indicies_g = torch.stack([lower, upper], dim=-1)

    matched_shape = [indicies_g.shape[0], indicies_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indicies_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indicies_g)
    
    denominator = cdf_g[..., 1] - cdf_g[..., 0]
    denominator = torch.where(denominator < 1e-6, torch.ones_like(denominator), denominator)
    t = (u - cdf_g[..., 0]) / denominator
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples