import os
import csv
import time
import subprocess
from PIL import Image
import argparse
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def get_gpu_usage():
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 ** 3  # Convert bytes to gigabytes

def get_cpu_memory_usage():
    return psutil.virtual_memory().used / 1024 ** 3  # Convert bytes to gigabytes

def get_folder_size_in_gb(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / 1024 ** 3  # Convert bytes to gigabytes

def train_gaussian_splatting(args):
    scenes_dir = args.scenes_root
    downsample_suffix = f"downsample{args.downsample}"
    output_file = os.path.join("results", "benchmarks", "train_benchmarks.csv")
    script_name = "segment_3d_gaussians/train_scene.py"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(scenes_dir):
        print(f"Directory {scenes_dir} does not exist.")
        return

    scenes = [d for d in os.listdir(scenes_dir) if d.endswith(downsample_suffix) and os.path.isdir(os.path.join(scenes_dir, d))]

    if not scenes:
        print(f"No scenes found ending with '{downsample_suffix}'.")
        return

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'approach', 'script', 'scene', 'time', 'image_count', 'image_width', 'image_height', 'scene_size',
                'avg_gpu_usage', 'max_gpu_usage', 'avg_cpu_memory_usage', 'max_cpu_memory_usage', 
                'avg_combined_usage', 'max_combined_usage'
            ])

        nvmlInit()  # Initialize NVML

        for scene in scenes:
            scene_path = os.path.join(scenes_dir, scene)
            input_path = os.path.join(scene_path, "input")
            model_path = os.path.join("results", "splatting_models", scene)

            if not os.path.exists(input_path):
                print(f"Input directory {input_path} does not exist for scene {scene}.")
                continue

            images = [img for img in os.listdir(input_path) if img.endswith(".jpg")]

            if not images:
                print(f"No images found in {input_path} for scene {scene}.")
                continue

            image_path = os.path.join(input_path, images[0])
            image_width, image_height = get_image_dimensions(image_path)
            image_count = len(images)
            scene_size = get_folder_size_in_gb(input_path)

            start_time = time.time()
            gpu_usages = []
            cpu_memory_usages = []
            combined_usages = []

            process = subprocess.Popen(["python", script_name, "-s", scene_path, "-m", model_path])

            while process.poll() is None:
                gpu_usage = get_gpu_usage()
                cpu_memory_usage = get_cpu_memory_usage()
                combined_usage = gpu_usage + cpu_memory_usage

                gpu_usages.append(gpu_usage)
                cpu_memory_usages.append(cpu_memory_usage)
                combined_usages.append(combined_usage)

                time.sleep(1)

            process.wait()  # Wait for the process to complete
            end_time = time.time()
            elapsed_time = end_time - start_time

            avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
            max_gpu_usage = max(gpu_usages)
            avg_cpu_memory_usage = sum(cpu_memory_usages) / len(cpu_memory_usages)
            max_cpu_memory_usage = max(cpu_memory_usages)
            avg_combined_usage = sum(combined_usages) / len(combined_usages)
            max_combined_usage = max(combined_usages)

            writer.writerow([
                'saga', 'train_scene.py', scene, elapsed_time, image_count, image_width, image_height, scene_size,
                avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage
            ])
            print(f"Processed scene {scene}: {elapsed_time:.2f} seconds, "
                  f"Avg GPU: {avg_gpu_usage:.2f} GB, Max GPU: {max_gpu_usage:.2f} GB, "
                  f"Avg CPU Memory: {avg_cpu_memory_usage:.2f} GB, Max CPU Memory: {max_cpu_memory_usage:.2f} GB, "
                  f"Avg Combined: {avg_combined_usage:.2f} GB, Max Combined: {max_combined_usage:.2f} GB, "
                  f"Scene Size: {scene_size:.2f} GB")

        nvmlShutdown()  # Shutdown NVML

def train_opennerf():
    """
    Function for opennerf training

    Args:
        --scenes-root   /home/luca_luis/adl4cv/nerf_segmentation/data/nerfstudio
        --downsample    Only valid arguments atm is 0
        --downscale     Only valid arguments are 0, 2, 4, 8
    """
    scenes_dir = args.scenes_root
    if args.downsample == "0":
        downsample_suffix = ""
    else:
        downsample_suffix = f"_{args.downsample}"
    downscale_suffix = ""
    output_file = os.path.join("results", "benchmarks", "train_benchmarks.csv")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(scenes_dir):
        print(f"Directory {scenes_dir} does not exist.")
        return

    scenes = [d for d in os.listdir(scenes_dir) if d.endswith(downsample_suffix) and os.path.isdir(os.path.join(scenes_dir, d))]

    if not scenes:
        print(f"No scenes found ending with '{downsample_suffix}'.")
        return

    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'approach', 'script', 'scene', 'time', 'image_count', 'image_width', 'image_height', 'scene_size',
                'avg_gpu_usage', 'max_gpu_usage', 'avg_cpu_memory_usage', 'max_cpu_memory_usage', 
                'avg_combined_usage', 'max_combined_usage'
            ])

        nvmlInit()  # Initialize NVML

        for scene in scenes:
            scene_path = os.path.join(scenes_dir, scene)
            input_path = os.path.join(scene_path, "images", downscale_suffix)
            output_path = os.path.join("results", "opennerf_outputs", scene)

            if not os.path.exists(input_path):
                print(f"Input directory {input_path} does not exist for scene {scene}.")
                continue

            images = [img for img in os.listdir(input_path) if img.endswith(".jpg")]

            if not images:
                print(f"No images found in {input_path} for scene {scene}.")
                continue

            image_path = os.path.join(input_path, images[0])
            image_width, image_height = get_image_dimensions(image_path)
            image_count = len(images)
            scene_size = get_folder_size_in_gb(input_path)

            start_time = time.time()
            gpu_usages = []
            cpu_memory_usages = []
            combined_usages = []

            process = subprocess.Popen(
                ["/home/luca_luis/anaconda3/envs/opennerf/bin/python",
                "/home/luca_luis/anaconda3/envs/opennerf/lib/python3.10/site-packages/nerfstudio/scripts/train.py",
                f"opennerf",
                f"--vis=wandb",  # viewer+wandb
                f"--data={scene_path}",
                f"--output-dir={output_path}",
                f"--timestamp={scene}"])

            while process.poll() is None:
                gpu_usage = get_gpu_usage()
                cpu_memory_usage = get_cpu_memory_usage()
                combined_usage = gpu_usage + cpu_memory_usage

                gpu_usages.append(gpu_usage)
                cpu_memory_usages.append(cpu_memory_usage)
                combined_usages.append(combined_usage)

                time.sleep(1)

            process.wait()  # Wait for the process to complete
            end_time = time.time()
            elapsed_time = end_time - start_time

            avg_gpu_usage = sum(gpu_usages) / len(gpu_usages)
            max_gpu_usage = max(gpu_usages)
            avg_cpu_memory_usage = sum(cpu_memory_usages) / len(cpu_memory_usages)
            max_cpu_memory_usage = max(cpu_memory_usages)
            avg_combined_usage = sum(combined_usages) / len(combined_usages)
            max_combined_usage = max(combined_usages)

            writer.writerow([
                'opennerf', 'nerfstudio/train.py', scene, elapsed_time, image_count, image_width, image_height, scene_size,
                avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage
            ])
            print(f"Processed scene {scene}: {elapsed_time:.2f} seconds, "
                  f"Avg GPU: {avg_gpu_usage:.2f} GB, Max GPU: {max_gpu_usage:.2f} GB, "
                  f"Avg CPU Memory: {avg_cpu_memory_usage:.2f} GB, Max CPU Memory: {max_cpu_memory_usage:.2f} GB, "
                  f"Avg Combined: {avg_combined_usage:.2f} GB, Max Combined: {max_combined_usage:.2f} GB, "
                  f"Scene Size: {scene_size:.2f} GB")

        nvmlShutdown()  # Shutdown NVML

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for training scenes.")
    parser.add_argument('--scenes_root', required=True, help="The path to the folder containing the scenes directory.")
    parser.add_argument('--downsample', required=True, help="The downsampling factor to look for in scene names.")
    parser.add_argument('--gaussian_splatting', action='store_true', help="Run the script for Gaussian Splatting.")
    parser.add_argument('--opennerf', action='store_true', help="Run the script for OpenNeRF.")
    args = parser.parse_args()

    if args.gaussian_splatting and args.opennerf:
        raise ValueError("Please select either --gaussian_splatting or --opennerf, not both.")
    elif not args.gaussian_splatting and not args.opennerf:
        raise ValueError("Please select either --gaussian_splatting or --opennerf.")

    if args.gaussian_splatting:
        train_gaussian_splatting(args)
    elif args.opennerf:
        train_opennerf()
