import os
import json
import numpy as np
import cv2
import pycolmap
import shutil
import rich
from rich.progress import Progress
import torch
from torch.utils.data import Dataset
from helpers import *


class NeRFDataset:
    def __init__(self, dataset_path: str):
        if "nerf_synthetic" in dataset_path:
            self._load_synthetic(dataset_path)
        elif "nerf_llff_data" in dataset_path:
            self._load_llff(dataset_path)
        elif "nerf_real_360" in dataset_path:
            self._load_real_360(dataset_path)
        else:
            raise ValueError(f"No dataset loader found for {dataset_path}")

    def _load_synthetic(self, dataset_path: str):
        rich.print(f"Loading synthetic dataset from {dataset_path}")

        self.data = {"type": "synthetic"}

        for split in ["train", "val", "test"]:
            with open(os.path.join(dataset_path, f"transforms_{split}.json")) as f:
                tf_data = json.load(f)

            self.data["camera_angle_x"] = float(tf_data["camera_angle_x"])
            
            paths = []
            rotations = []
            transforms = []

            file_endings = {}
            for file in os.listdir(os.path.join(dataset_path, split)):
                file_endings[file.split(".")[0]] = file

            for frame in tf_data["frames"]:
                paths.append(os.path.join(dataset_path, split, file_endings[frame["file_path"].split("/")[-1]]))
                rotations.append(float(frame["rotation"]))
                transforms.append(np.array(frame["transform_matrix"], np.float32))
            
            self.data[split] = {
                "path": paths,
                "rotation": rotations,
                "transform": transforms,
                "len": len(tf_data["frames"])
            }
        
        colmap_path = os.path.join(dataset_path, "colmap")
        if not os.path.exists(colmap_path):
            os.mkdir(colmap_path)
        
        valid_reconstruction = True

        try:
            sparse_path = os.path.join(colmap_path, "sparse")
            sparse0_path = os.path.join(sparse_path, "0")
            reconstruction = pycolmap.Reconstruction(sparse0_path)
        except:
            valid_reconstruction = False
        
        if not valid_reconstruction:
            shutil.rmtree(colmap_path)
            os.mkdir(colmap_path)

            database_path = os.path.join(colmap_path, "database.db")

            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.use_gpu = True

            image_path = os.path.dirname(self.data["train"]["path"][0])

            pycolmap.extract_features(
                database_path,
                image_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model="SIMPLE_PINHOLE",
                sift_options=sift_options
            )

            sift_options = pycolmap.SiftMatchingOptions()
            sift_options.use_gpu = True

            pycolmap.match_exhaustive(database_path, sift_options)

            mapper_options = pycolmap.IncrementalPipelineOptions()
            mapper_options.ba_refine_focal_length = True
            mapper_options.ba_refine_principal_point = True
            mapper_options.ba_refine_extra_params = False

            pycolmap.incremental_mapping(
                database_path,
                image_path,
                sparse_path,
                mapper_options
            )
        
            reconstruction = pycolmap.Reconstruction(sparse0_path)

        rich.print(f"Loaded reconstruction from {sparse0_path}")

        cam = reconstruction.cameras[1]
        rich.print(f"Found Camera Intrinsics:")
        rich.print(f"  fx", cam.focal_length_x)
        rich.print(f"  fy", cam.focal_length_y)
        rich.print(f"  px", cam.principal_point_x)
        rich.print(f"  py", cam.principal_point_y)
        
        self.data["K"] = np.array([
            [cam.focal_length_x, 0, cam.principal_point_x],
            [0, cam.focal_length_y, cam.principal_point_y],
            [0, 0, 1]
        ], np.float32)
        

    def _load_llff(self, dataset_path: str):
        raise NotImplementedError("TODO")

    def _load_real_360(self, dataset_path: str):
        raise NotImplementedError("TODO")
    
    def load_image(self, image_path: str, scale: float=None):
        img = cv2.imread(image_path)
        if len(img.shape) != 3:
            raise ValueError(f"unknown image shape {img.shape} -- should be a tuple of length 3")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        match img.dtype:
            case np.uint8:
                img = img.astype(np.float32) / np.iinfo(np.uint8).max
            case np.uint16:
                img = img.astype(np.float32) / np.iinfo(np.uint16).max
            case np.float32 | np.float64:
                pass
            case _:
                raise ValueError("unknown image data type")
        if scale is not None:
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        return img
    

class NeRFDataLoaderDataset(Dataset):
    def __init__(self, dataset: NeRFDataset, split: str, scale: float=None):
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.scale = scale

        self.cached_K = []
        self.cached_ray_origins = []
        self.cached_ray_dirs = []
        self.cached_images = []

        with Progress() as p:
            task = p.add_task("Caching dataset rays + images ...", total=len(self))
            for i in range(len(self)):
                scale_matrix = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]
                ], np.float32)
                self.cached_K.append(scale_matrix.dot(self.dataset.data["K"]))
                c2w = self.dataset.data[self.split]["transform"][i]
                img = self.dataset.load_image(self.dataset.data[self.split]["path"][i], self.scale)
                height, width = img.shape[:2]
                ray_origins, ray_dirs = get_rays(height, width, self.cached_K[-1], c2w)
                self.cached_ray_origins.append(ray_origins)
                self.cached_ray_dirs.append(ray_dirs)
                self.cached_images.append(img)
                p.update(task, advance=1)

    def __len__(self):
        return self.dataset.data[self.split]["len"]

    def __getitem__(self, idx):
        K = torch.from_numpy(self.cached_K[idx])
        c2w = torch.from_numpy(self.dataset.data[self.split]["transform"][idx])
        ray_origins = torch.from_numpy(self.cached_ray_origins[idx].copy())
        ray_dirs = torch.from_numpy(self.cached_ray_dirs[idx])
        img = torch.from_numpy(self.cached_images[idx])
        return K, c2w, ray_origins, ray_dirs, img


import time
class FastWeightedSampler:
    def __init__(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        if probs.sum() == 0:
            raise ValueError("Probabilities sum to zero")
        self.cdf = np.cumsum(probs)  # Precompute CDF once
        self.cdf /= self.cdf[-1]     # Normalize to [0, 1]

    def sample(self, size):
        r = np.random.random(size)
        return np.searchsorted(self.cdf, r)

class NeRFRayDataLoader:
    def __init__(self, dataset: NeRFDataset, split: str):
        self.dataset = dataset
        self.split = split

        self.cached_dims = []
        self.cached_pixels = []
        self.cached_ray_origins = []
        self.cached_ray_dirs = []
        
        with Progress() as p:
            task = p.add_task("Loading rays + images ...", total=len(self))

            for i in range(len(self)):
                K = self.dataset.data["K"]
                c2w = self.dataset.data[self.split]["transform"][i]
                img = self.dataset.load_image(self.dataset.data[self.split]["path"][i])
                height, width = img.shape[:2]
                ray_origins, ray_dirs = get_rays(height, width, K, c2w)

                pixels = img.reshape(-1, img.shape[-1])
                ray_origins = ray_origins.reshape(-1, ray_origins.shape[-1])
                ray_dirs = ray_dirs.reshape(-1, ray_origins.shape[-1])

                self.cached_dims.append((height, width))
                self.cached_pixels.append(pixels)
                self.cached_ray_origins.append(ray_origins)
                self.cached_ray_dirs.append(ray_dirs)
                p.update(task, advance=1)
        
        self.cached_dims = np.concat(self.cached_dims, axis=0)
        self.cached_pixels = np.concat(self.cached_pixels, axis=0)
        self.cached_ray_origins = np.concat(self.cached_ray_origins, axis=0)
        self.cached_ray_dirs = np.concat(self.cached_ray_dirs, axis=0)
        mask = (self.cached_pixels > 0.05).any(axis=1)
        self.prob = (0.75 * mask) + 0.25 * (1 - mask)
        self.prob = self.prob / self.prob.sum()
        self.sampler = FastWeightedSampler(self.prob)
        
    def __len__(self):
        return self.dataset.data[self.split]["len"]
    
    def sample(self, batch_size):
        indicies = self.sampler.sample(batch_size)
        pixels = torch.from_numpy(self.cached_pixels[indicies])
        ray_origins = torch.from_numpy(self.cached_ray_origins[indicies])
        ray_dirs = torch.from_numpy(self.cached_ray_dirs[indicies])
        return pixels, ray_origins, ray_dirs
    
    def load_image(self, index, scale=None):
        K = self.dataset.data["K"]
        if scale is not None:
            scale_matrix = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]
                ], np.float32)
            K = scale_matrix.dot(K)
        
        c2w = self.dataset.data[self.split]["transform"][index]
        img = torch.from_numpy(self.dataset.load_image(self.dataset.data[self.split]["path"][index], scale))
        ray_origins, ray_dirs = get_rays(*img.shape[:2], K, c2w)
        ray_origins = torch.from_numpy(ray_origins.copy())
        ray_dirs = torch.from_numpy(ray_dirs.copy())
        K = torch.from_numpy(K)
        c2w = torch.from_numpy(c2w)
        return K, c2w, ray_origins, ray_dirs, img
    


if __name__ == "__main__":
    ds = NeRFRayDataLoader(
        NeRFDataset(os.path.join("./dataset", "nerf_synthetic", "drums")),
        "train",
    )

    while True:
        pixels, ray_origins, ray_dirs = ds.sample(4096)
        print(pixels.shape, ray_origins.shape, ray_dirs.shape)