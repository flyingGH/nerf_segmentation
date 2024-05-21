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


def preprocess_camera_intrinsics(json_path, binary_path):
    with open(json_path, 'r') as f:
        cam_params = json.load(f)

    w = cam_params['camera']['w']
    h = cam_params['camera']['h']
    fx = cam_params['camera']['fx']
    fy = cam_params['camera']['fy']
    cx = cam_params['camera']['cx']
    cy = cam_params['camera']['cy']
    scale = cam_params['camera']['scale']  # not used in the binary format

    params = [fx, fy, cx, cy] # cam_intrinsics

    camera_id = 1 # single camera
    model_id = 1  # pinhole model

    model_name = CAMERA_MODEL_IDS[model_id].model_name
    num_params = CAMERA_MODEL_IDS[model_id].num_params

    camera = Camera(id=camera_id, model=model_name, width=w, height=h, params=np.array(params))
    cameras = {camera_id: camera}

    with open(binary_path, "wb") as fid:
        fid.write(struct.pack("Q", len(cameras))) # n_cameras

        for _, cam in cameras.items():
            fid.write(struct.pack("iiQQ", cam.id, model_id, cam.width, cam.height)) # cam_properties
            fid.write(struct.pack("d" * num_params, *cam.params)) # cam_params

    print("[+] Successfully converted camera intrinsics to binary format")

def process_txt(filename):
   """
   Process a text file and return a list of lines.

   Args:
      filename (str): The path to the text file.

   Returns:
      list: A list of lines from the text file.
   """
   with open(filename) as file:
      lines = file.readlines()
      lines = [line.rstrip() for line in lines]
   return lines

def parse_matrix_string(matrix_string):
   """
   Parses a matrix string and returns a 4x4 numpy array.

   Args:
      matrix_string (str): A string representing a matrix with space-separated values.

   Returns:
      numpy.ndarray: A 4x4 numpy array representing the parsed matrix.

   Raises:
      ValueError: If the matrix string does not contain 16 space-separated values.
   """
   matrix_values = list(map(float, matrix_string.split()))
   if len(matrix_values) != 16:
      raise ValueError("Invalid matrix string. Expected 16 space-separated values.")
   matrix = np.array(matrix_values).reshape((4, 4))
   return matrix

def matrix_to_quat_trans(mat):
   """
   Convert a 4x4 matrix to quaternion and translation vector.

   Args:
      mat (numpy.ndarray): Input matrix of shape (4, 4).

   Returns:
      tuple: A tuple containing the quaternion and translation vector.

   Raises:
      AssertionError: If the input matrix is not of shape (4, 4).
   """

   assert mat.shape == (4, 4), "Input matrix must be 4x4"

   rotation_matrix = mat[:3, :3]
   translation_vector = mat[:3, 3]

   r = R.from_matrix(rotation_matrix)
   quaternion = r.as_quat()

   return quaternion, translation_vector
