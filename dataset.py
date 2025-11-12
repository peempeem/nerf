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


trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


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


            w, h = 800, 800
            focal = 0.5 * w / np.tan(0.5 * tf_data["camera_angle_x"])

            self.data["K"] = np.array([
                [focal, 0, w / 2],
                [0, focal, h / 2],
                [0, 0, 1]
            ], np.float32)
            
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

            transforms = np.asarray(transforms)
            
            self.data[split] = {
                "path": paths,
                "rotation": rotations,
                "transform": transforms,
                "len": len(tf_data["frames"]),
                "bound_norm_scale": np.abs(transforms[:, :3, 3]).max()
            }
        

    def _load_llff(self, dataset_path: str, image_dir="images_8"):
        rich.print(f"Loading llff dataset from {dataset_path}")

        self.data = {"type": "llff"}

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

        image_path = os.path.join(dataset_path, image_dir)
        
        if not valid_reconstruction:
            shutil.rmtree(colmap_path)
            os.mkdir(colmap_path)

            database_path = os.path.join(colmap_path, "database.db")

            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_image_size = 1600
            sift_options.max_num_features = 4096
            sift_options.use_gpu = True

            pycolmap.extract_features(
                database_path,
                image_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model="SIMPLE_RADIAL",
                sift_options=sift_options,
                device=pycolmap.Device.auto
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
        
        self.data["focal"] = cam.focal_length_x
        self.data["K"] = np.array([
            [cam.focal_length_x, 0, cam.principal_point_x],
            [0, cam.focal_length_y, cam.principal_point_y],
            [0, 0, 1]
        ], np.float32)

        paths = []
        transforms = []
        for image in reconstruction.images.values():
            paths.append(os.path.join(image_path, image.name))
            c2w = image.cam_from_world().inverse().matrix()
            c2w = np.concat([c2w, np.array([[0, 0, 0, 1]])], axis=0)
            transforms.append(c2w)

        self.data["train"] = {
            "path": paths,
            "transform": np.asarray(transforms, np.float32),
            "len": len(transforms),
            "bound_norm_scale": 1
        }   


    def _load_real_360(self, dataset_path: str):
        raise NotImplementedError("TODO")
    
    def load_image(self, image_path: str, scale: float=None):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img.reshape((*img.shape, 1))
        elif len(img.shape) != 3:
            raise ValueError(f"unknown image shape {img.shape} -- should be a tuple of length 3")
        else:
            match img.shape[-1]:
                case 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
                case 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                case 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                case _:
                    raise ValueError(f"unknown # of image channels: {img.shape[-1]} -- should be 1, 3, or 4")

        match img.dtype:
            case np.uint8:
                img = img.astype(np.float32) / np.iinfo(np.uint8).max
            case np.uint16:
                img = img.astype(np.float32) / np.iinfo(np.uint16).max
            case np.float32 | np.float64:
                pass
            case _:
                raise ValueError("unknown image data type")
        
        rgb = img[..., :3]
        alpha = img[..., 3:4]
        white = np.full_like(rgb, 1)

        img = (alpha * rgb + (1 - alpha) * white)
        
        if scale is not None:
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        return img


class NeRFRayDataLoader:
    def __init__(self, dataset: NeRFDataset, split: str):
        self.dataset = dataset
        self.split = split

        self.cached_rgb = []
        self.cached_ray_origins = []
        self.cached_ray_dirs = []
        
        with Progress() as p:
            task = p.add_task("Loading rays + images ...", total=len(self))

            for i in range(len(self)):
                K, c2w, ray_origins, ray_dirs, img = self.load_image(i)
                img = img.numpy()

                self.cached_rgb.append(img.reshape(-1, 3))
                self.cached_ray_origins.append(ray_origins.numpy().reshape(-1, ray_origins.shape[-1]))
                self.cached_ray_dirs.append(ray_dirs.numpy().reshape(-1, ray_dirs.shape[-1]))
                p.update(task, advance=1)
        
        self.cached_rgb = np.concat(self.cached_rgb, axis=0)
        self.cached_ray_origins = np.concat(self.cached_ray_origins, axis=0)
        self.cached_ray_dirs = np.concat(self.cached_ray_dirs, axis=0)
        
    def __len__(self):
        return self.dataset.data[self.split]["len"]
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.cached_rgb), batch_size, replace=True)
        rgbs = torch.from_numpy(self.cached_rgb[indices])
        ray_origins = torch.from_numpy(self.cached_ray_origins[indices])
        ray_dirs = torch.from_numpy(self.cached_ray_dirs[indices])
        return rgbs, ray_origins, ray_dirs
    
    def load_image(self, index, scale=None, bound_norm_scale=None, near=1):
        K = self.dataset.data["K"]
        if scale is not None:
            scale_matrix = np.array([
                    [scale, 0, 0],
                    [0, scale, 0],
                    [0, 0, 1]
                ], np.float32)
            K = scale_matrix.dot(K)
        
        c2w = self.dataset.data[self.split]["transform"][index].copy()
        img = torch.from_numpy(self.dataset.load_image(self.dataset.data[self.split]["path"][index], scale))
        c2w[:3, 3] /= self.dataset.data[self.split]["bound_norm_scale"] if bound_norm_scale is None else bound_norm_scale
        ray_origins, ray_dirs = get_rays(*img.shape[:2], K, c2w)
        if self.dataset.data["type"] != "synthetic":
            ray_origins, ray_dirs = ndc_rays(*img.shape[:2], self.dataset.data["focal"], near, ray_origins, ray_dirs)
        ray_origins = torch.from_numpy(ray_origins.copy())
        ray_dirs = torch.from_numpy(ray_dirs.copy())
        K = torch.from_numpy(K)
        c2w = torch.from_numpy(c2w)
        return K, c2w, ray_origins, ray_dirs, img
    


if __name__ == "__main__":
    ds = NeRFRayDataLoader(
        NeRFDataset(os.path.join("./dataset", "nerf_llff_data", "fern")),
        "train",
    )

    while True:
        K, c2w, ray_origins, ray_dirs, img = ds.load_image(0)
        print(c2w)