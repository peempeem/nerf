import os
import json
import numpy as np
import cv2


class Dataset:
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
                transforms.append(np.array(frame["transform_matrix"], np.float64))
            
            self.data[split] = {
                "path": paths,
                "rotation": rotations,
                "transform": transforms,
                "len": len(tf_data["frames"])
            }
        
        colmap_path = os.path.join(dataset_path, "colmap")
        if not os.path.exists():
            os.mkdir(colmap_path)
        

    def _load_llff(self, dataset_path: str):
        raise NotImplementedError("TODO")

    def _load_real_360(self, dataset_path: str):
        raise NotImplementedError("TODO")
    
    def load_image(self, image_path: str):
        img = cv2.imread(image_path)
        if len(img.shape) != 3:
            raise ValueError(f"unknown image shape {img.shape} -- should be a tuple of length 3")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        match img.dtype:
            case np.uint8:
                img = img.astype(np.float32) / np.iinfo(np.uint8).max
            case np.uint16:
                img = img.astype(np.float32) / np.iinfo(np.uint8).max
            case np.float32 | np.float64:
                pass
            case _:
                raise ValueError("unknown image data type")
        return img


if __name__ == "__main__":
    dataset = Dataset(os.path.join("dataset", "nerf_synthetic", "drums"))

    split = "train"
    for i in range(dataset.data[split]["len"]):
        img = dataset.load_image(dataset.data[split]["path"][i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
        cv2.waitKey(100)