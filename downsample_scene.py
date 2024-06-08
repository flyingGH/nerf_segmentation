import os
import argparse
import shutil
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Downsample a dataset by copying every nth file.')
    parser.add_argument('scene_root', type=str, help='Path to the scene root directory')
    parser.add_argument('--downsample', type=int, required=True, help='Downsample factor')
    return parser.parse_args()

def check_directories(scene_root):
    input_dir = os.path.join(scene_root, 'input')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Subdirectory 'input' is missing in the scene root: {scene_root}")
    return input_dir

def create_downsampled_directory(scene_root, downsample_factor):
    scene_name = os.path.basename(os.path.normpath(scene_root))
    new_scene_name = f"{scene_name}_downsample{downsample_factor}"
    new_scene_root = os.path.join(os.path.dirname(scene_root), new_scene_name)
    os.makedirs(os.path.join(new_scene_root, 'input'), exist_ok=True)
    return new_scene_root

def downsample_directory(input_dir, new_input_dir, downsample_factor):
    files = sorted(os.listdir(input_dir))
    downsampled_files = files[::downsample_factor]

    for file in tqdm(downsampled_files, desc='Copying files'):
        src_file = os.path.join(input_dir, file)
        dst_file = os.path.join(new_input_dir, file)
        shutil.copy2(src_file, dst_file)

def main():
    """
    Downsample a dataset by copying every nth file from the original directory to a new directory.
    
    Usage:
        python downsample_scene.py <scene_root> --downsample <downsample_factor>
    
    Arguments:
        scene_root: Path to the scene root directory which contains the 'input' subdirectory.
        --downsample: Integer value indicating the downsampling factor (every nth file will be copied).
    """
    args = parse_arguments()

    try:
        input_dir = check_directories(args.scene_root)
    except FileNotFoundError as e:
        print(e)
        return

    new_scene_root = create_downsampled_directory(args.scene_root, args.downsample)
    new_input_dir = os.path.join(new_scene_root, 'input')

    downsample_directory(input_dir, new_input_dir, args.downsample)
    print(f"Downsampling complete. New directory created at: {new_scene_root}")

if __name__ == "__main__":
    main()
