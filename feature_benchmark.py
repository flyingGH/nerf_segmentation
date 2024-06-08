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

def run_script(command):
    start_time = time.time()
    gpu_usages = []
    cpu_memory_usages = []
    combined_usages = []

    process = subprocess.Popen(command, shell=True)

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

    return elapsed_time, avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage

def benchmark_features(args):
    scenes_dir = args.scenes_root
    downsample_suffix = f"downsample{args.downsample}"
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
            input_path = os.path.join(scene_path, "input")
            model_path = os.path.join(args.model_root, scene)

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

            scripts = [
                f"python segment_3d_gaussians/extract_segment_everything_masks.py --image_root {scene_path} --sam_checkpoint_path {args.sam_path}",
                f"python segment_3d_gaussians/get_scale.py --image_root {scene_path} --model_path {model_path}",
                f"python segment_3d_gaussians/get_clip_features.py --image_root {scene_path} --clip_path {args.clip_path}",
                f"python segment_3d_gaussians/train_contrastive_feature.py -m {model_path} --iterations 10000 --num_sampled_rays 1000"
            ]

            for script in scripts:
                elapsed_time, avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage = run_script(script)

                writer.writerow([
                    'saga', script.split()[1], scene, elapsed_time, image_count, image_width, image_height, scene_size,
                    avg_gpu_usage, max_gpu_usage, avg_cpu_memory_usage, max_cpu_memory_usage, avg_combined_usage, max_combined_usage
                ])
                print(f"Processed scene {scene}: {elapsed_time:.2f} seconds, "
                      f"Avg GPU: {avg_gpu_usage:.2f} GB, Max GPU: {max_gpu_usage:.2f} GB, "
                      f"Avg CPU Memory: {avg_cpu_memory_usage:.2f} GB, Max CPU Memory: {max_cpu_memory_usage:.2f} GB, "
                      f"Avg Combined: {avg_combined_usage:.2f} GB, Max Combined: {max_combined_usage:.2f} GB, "
                      f"Scene Size: {scene_size:.2f} GB")

        nvmlShutdown()  # Shutdown NVML

if __name__ == "__main__":
    """
    Benchmark script for feature extraction and processing scenes.
    
    Usage:
    python feature_benchmark.py --scenes_root <path_to_scenes> --downsample <downsample_factor> --sam_path <path_to_sam> --model_root <path_to_models> --clip_path <path_to_clip>
    """
    parser = argparse.ArgumentParser(description="Benchmark script for feature extraction and processing scenes.")
    parser.add_argument('--scenes_root', required=True, help="The path to the folder containing the scenes directory.")
    parser.add_argument('--downsample', required=True, help="The downsampling factor to look for in scene names.")
    parser.add_argument('--sam_path', required=True, help="Path to the SAM checkpoint.")
    parser.add_argument('--model_root', required=True, help="Root directory where models are stored.")
    parser.add_argument('--clip_path', required=True, help="Path to the CLIP checkpoint.")
    args = parser.parse_args()

    benchmark_features(args)
