from huggingface_hub.utils import validate_repo_id, HfHubHTTPError
from GoogleDrive.DriveConnections import DriveConnection
from urllib.parse import urlparse, unquote
from huggingface_hub import HfFileSystem
from IPython.utils import capture
from subprocess import getoutput
from pathlib import Path

import subprocess
import requests
import zipfile
import shutil
import gdown
import torch
import json
import time
import glob
import ast
import sys
import re
import os

root_dir          = "/root/content"
drive_dir         = os.path.join(root_dir, "drive/MyDrive")
deps_dir          = os.path.join(root_dir, "deps")
repo_dir          = os.path.join(root_dir, "kohya-trainer")
training_dir      = os.path.join(root_dir, "LoRA")
pretrained_model  = os.path.join(root_dir, "pretrained_model")
vae_dir           = os.path.join(root_dir, "vae")
lora_dir          = os.path.join(root_dir, "network_weight")
repositories_dir  = os.path.join(root_dir, "repositories")
config_dir        = os.path.join(training_dir, "config")
tools_dir         = os.path.join(repo_dir, "tools")
finetune_dir      = os.path.join(repo_dir, "finetune")
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")

HUGGINGFACE_TOKEN     = "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" #@param {type: "string"}
LOAD_DIFFUSERS_MODEL  = True #@param {type: "boolean"}
SDXL_MODEL_URL        = "stabilityai/stable-diffusion-xl-base-1.0" # @param ["gsdf/CounterfeitXL", "Linaqruf/animagine-xl", "stabilityai/stable-diffusion-xl-base-1.0", "PASTE MODEL URL OR GDRIVE PATH HERE"] {allow-input: true}
SDXL_VAE_URL          = "Original VAE" # @param ["None", "Original VAE", "FP16 VAE", "PASTE VAE URL OR GDRIVE PATH HERE"] {allow-input: true}


HUGGINGFACE_TOKEN     = "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" #@param {type: "string"}
LOAD_DIFFUSERS_MODEL  = True #@param {type: "boolean"}
SDXL_MODEL_URL        = "stabilityai/stable-diffusion-xl-base-1.0" # @param ["gsdf/CounterfeitXL", "Linaqruf/animagine-xl", "stabilityai/stable-diffusion-xl-base-1.0", "PASTE MODEL URL OR GDRIVE PATH HERE"] {allow-input: true}
SDXL_VAE_URL          = "Original VAE" # @param ["None", "Original VAE", "FP16 VAE", "PASTE VAE URL OR GDRIVE PATH HERE"] {allow-input: true}

MODEL_URLS = {
    "gsdf/CounterfeitXL"        : "https://huggingface.co/gsdf/CounterfeitXL/resolve/main/CounterfeitXL_%CE%B2.safetensors",
    "Linaqruf/animagine-xl"   : "https://huggingface.co/Linaqruf/animagine-xl/resolve/main/animagine-xl.safetensors",
    "stabilityai/stable-diffusion-xl-base-1.0" : "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
}
VAE_URLS = {
    "None"                    : "",
    "Original VAE"           : "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
    "FP16 VAE"           : "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors"
}

SDXL_MODEL_URL = MODEL_URLS.get(SDXL_MODEL_URL, SDXL_MODEL_URL)
SDXL_VAE_URL = VAE_URLS.get(SDXL_VAE_URL, SDXL_VAE_URL)

def get_filename(url):
    if any(url.endswith(ext) for ext in [".ckpt", ".safetensors", ".pt", ".pth"]):
        return os.path.basename(url)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    if 'content-disposition' in response.headers:
        filename = re.findall('filename="?([^"]+)"?', response.headers['content-disposition'])[0]
    else:
        filename = unquote(os.path.basename(urlparse(url).path))

    return filename

def aria2_download(dir, filename, url):
    user_header = f"Authorization: Bearer {HUGGINGFACE_TOKEN}"
    aria2_args = [
        "aria2c",
        "--console-log-level=error",
        "--summary-interval=10",
        f"--header={user_header}" if "huggingface.co" in url else "",
        "--continue=true",
        "--max-connection-per-server=16",
        "--min-split-size=1M",
        "--split=16",
        f"--dir={dir}",
        f"--out={filename}",
        url
    ]
    subprocess.run(aria2_args)

def download(url, dst):
    print(f"Starting downloading from {url}")
    filename = get_filename(url)
    filepath = os.path.join(dst, filename)

    if "drive.google.com" in url:
        gdown.download(url, filepath, quiet=False)
    else:
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/ resolve/")
        aria2_download(dst, filename, url)

    print(f"Download finished: {filepath}")
    return filepath

def all_folders_present(base_model_url, sub_folders):
    fs = HfFileSystem()
    existing_folders = set(fs.ls(base_model_url, detail=False))

    for folder in sub_folders:
        full_folder_path = f"{base_model_url}/{folder}"
        if full_folder_path not in existing_folders:
            return False
    return True

def get_total_ram_gb():
    with open('/proc/meminfo', 'r') as f:
        for line in f.readlines():
            if "MemTotal" in line:
                return int(line.split()[1]) / (1024**2)  # Convert to GB

def get_gpu_name():
    try:
        return subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader,nounits", shell=True).decode('ascii').strip()
    except:
        return None

def download_sdxl_model():
    global model_path, vae_path, LOAD_DIFFUSERS_MODEL

    model_path, vae_path = None, None

    required_sub_folders = [
        'scheduler',
        'text_encoder',
        'text_encoder_2',
        'tokenizer',
        'tokenizer_2',
        'unet',
        'vae',
    ]
    pretrained_model = "/root/content/pretrained_model/"
    download_targets = {
        "model": (SDXL_MODEL_URL, pretrained_model),
        "vae": (SDXL_VAE_URL, vae_dir),
    }

    total_ram = get_total_ram_gb()
    gpu_name = get_gpu_name()

    # Check hardware constraints
    if total_ram < 11 and gpu_name in ["Tesla T4", "Tesla V100"]:
        print("Attempt to load diffusers model instead due to hardware constraints.")
        if not LOAD_DIFFUSERS_MODEL:
            LOAD_DIFFUSERS_MODEL = True

    for target, (url, dst) in download_targets.items():
        if url and not url.startswith(f"PASTE {target.upper()} URL OR GDRIVE PATH HERE"):
            if target == "model" and LOAD_DIFFUSERS_MODEL:
                # Code for checking and handling diffusers model
                if 'huggingface.co' in url:
                    match = re.search(r'huggingface\.co/([^/]+)/([^/]+)', SDXL_MODEL_URL)
                    if match:
                        username = match.group(1)
                        model_name = match.group(2)
                        url = f"{username}/{model_name}"
                if all_folders_present(url, required_sub_folders):
                    print(f"Diffusers model is loaded : {url}")
                    print(required_sub_folders,"====++====+++")
                    model_path = url
                else:
                    print("Repository doesn't exist or no diffusers model detected.")

                    filepath = download(url, dst)  # Continue with the regular download
                    model_path = filepath
            else:
                filepath = download(url, dst)

                if target == "model":
                    model_path = filepath
                elif target == "vae":
                    vae_path = filepath

            print()

    if model_path:
        print(f"Selected model: {model_path}")

    if vae_path:
        print(f"Selected VAE: {vae_path}")

download_sdxl_model()

#@title ## **5.1. Inference**

import os
# %store -r

# @markdown ### Model Config
network_weights = "/root/content/drive/MyDrive/kohya-trainer/output/sdxl_lora/sdxl_lora.safetensors" #@param {type:'string'}
network_mul = 0.7 # @param {type:"slider", min:-1, max:1, step:0.05}
# @markdown ### Prompt Config
prompt = "a Vizsla dog sitting on a throne in a room, solo, sitting, indoors, cape, no humans, crown, red cape, throne" #@param {type:'string'}
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry" #@param {type:'string'}
output_path = "/root/content/tmp/" #@param {type:'string'}
resolution = "1024, 1024" # @param {type: "string"}
optimization = "scaled dot-product attention" # @param ["xformers", "scaled dot-product attention"]
conditional_resolution = "1024, 1024" # @param {type: "string"}
steps = 28 # @param {type: "number"}
sampler = "euler_a"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
scale = 7 # @param {type: "number"}
seed = -1 # @param {type: "number"}
images_per_prompt = 4 # @param {type: "number"}
batch_size = 1 # @param {type: "number"}
clip_skip = 2 # @param {type: "number"}

os.makedirs(output_path, exist_ok=True)

separators = ["*", "x", ","]

for separator in separators:
    if separator in resolution:
        width, height = [value.strip() for value in resolution.split(separator)]
        original_width, original_height = [value.strip() for value in conditional_resolution.split(separator)]
        break

network_config = {
    "network_module": "networks.lora",
    "network_weights": network_weights,
    "network_show_meta": True,
    "network_mul": network_mul,
}

config = {
    "prompt": prompt + " --n " + negative_prompt,
    "images_per_prompt": images_per_prompt,
    "outdir": output_path,
    "W": width,
    "H": height,
    "original_width": original_width,
    "original_height": original_height,
    "batch_size": batch_size,
    "vae_batch_size": 1,
    "no_half_vae": True,
    "steps": steps,
    "sampler": sampler,
    "scale": scale,
    "ckpt": model_path,
    "vae": vae_path,
    "seed": seed if seed > 0 else None,
    "fp16": True,
    "sdpa": True if optimization == "scaled dot-product attention" else False,
    "xformers": True if optimization == "xformers" else False,
    "opt_channels_last": True,
    "clip_skip": clip_skip,
    "max_embeddings_multiples": 3,
}

if network_weights != "":
    config.update(network_config)

args = ""
for k, v in config.items():
    if k.startswith("_"):
        args += f'"{v}" '
    elif isinstance(v, str):
        args += f'--{k}="{v}" '
    elif isinstance(v, bool) and v:
        args += f"--{k} "
    elif isinstance(v, float) and not isinstance(v, bool):
        args += f"--{k}={v} "
    elif isinstance(v, int) and not isinstance(v, bool):
        args += f"--{k}={v} "

os.chdir(repo_dir)
os.system(f"python sdxl_gen_img.py {args}")