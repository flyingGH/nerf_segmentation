import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import collections
import json
import os
import shutil
from tqdm import tqdm

# Camera Models taken from Gaussian Splatting repository
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def get_scenes(replica_path):
    all_items = os.listdir(replica_path)
    scene_names = [item for item in all_items if os.path.isdir(os.path.join(replica_path, item))]

    return scene_names

def extract_images(scene_path):
    results_path = os.path.join(scene_path, 'results')
    images_path = os.path.join(scene_path, 'images')

    os.makedirs(images_path, exist_ok=True)
    images = sorted([f for f in os.listdir(results_path) if f.startswith('frame') and f.endswith('.jpg')])

    with tqdm(total=len(images), desc=f'Extracting images', unit="file") as pbar:
        for img in images:
            src = os.path.join(results_path, img)
            dst = os.path.join(images_path, img)
            
            shutil.copy(src, dst)
            pbar.update(1)


def downsample_images(scene_path, downsampling_factor):

    images_path = os.path.join(scene_path, 'images')
    downsampled_path = os.path.join(scene_path, f'images_{downsampling_factor}')

    os.makedirs(downsampled_path, exist_ok=True)
    images = sorted([f for f in os.listdir(images_path) if f.startswith('frame') and f.endswith('.jpg')])

    with tqdm(total=len(images)//downsampling_factor, desc=f'Downsampling images', unit="file") as pbar:
        for i in range(0, len(images), downsampling_factor):
            src = os.path.join(images_path, images[i])
            dst = os.path.join(downsampled_path, images[i])

            shutil.copy(src, dst)
            pbar.update(1)

