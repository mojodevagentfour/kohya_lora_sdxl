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




# root_dir

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

for store in ["root_dir", "repo_dir", "training_dir", "pretrained_model", "vae_dir", "repositories_dir", "accelerate_config", "tools_dir", "finetune_dir", "config_dir"]:
    with capture.capture_output() as cap:
#         %store {store}
        del cap

repo_dict = {
    "qaneel/kohya-trainer (forked repo, stable, optimized for colab use)" : "https://github.com/qaneel/kohya-trainer",
    "kohya-ss/sd-scripts (original repo, latest update)"                    : "https://github.com/kohya-ss/sd-scripts",
}

repository        = "qaneel/kohya-trainer (forked repo, stable, optimized for colab use)" #@param ["qaneel/kohya-trainer (forked repo, stable, optimized for colab use)", "kohya-ss/sd-scripts (original repo, latest update)"] {allow-input: true}
repo_url          = repo_dict[repository]
branch            = "main"  # @param {type: "string"}
output_to_drive   = True  # @param {type: "boolean"}

def clone_repo(url, dir, branch):
    if not os.path.exists(dir):
       os.system(f"git clone -b {branch} {url} {dir}")

def mount_drive(dir):
    output_dir      = os.path.join(training_dir, "output")

    if output_to_drive:
        if not os.path.exists(drive_dir):
            os.makedirs(drive_dir)
        output_dir  = os.path.join(drive_dir, "kohya-trainer/output")

    return output_dir

def setup_directories():
    global output_dir

    output_dir= mount_drive(drive_dir)

    for dir in [training_dir, config_dir, pretrained_model, vae_dir, repositories_dir, output_dir]:
        os.makedirs(dir, exist_ok=True)

def pastebin_reader(id):
    if "pastebin.com" in id:
        url = id
        if 'raw' not in url:
                url = url.replace('pastebin.com', 'pastebin.com/raw')
    else:
        url = "https://pastebin.com/raw/" + id
    response = requests.get(url)
    response.raise_for_status()
    lines = response.text.split('\n')
    return lines

def install_repository():
    global infinite_image_browser_dir, voldy, discordia_archivum_dir

    _, voldy = pastebin_reader("kq6ZmHFU")[:2]

    infinite_image_browser_url  = f"https://github.com/zanllp/{voldy}-infinite-image-browsing.git"
    infinite_image_browser_dir  = os.path.join(repositories_dir, f"infinite-image-browsing")
    infinite_image_browser_deps = os.path.join(infinite_image_browser_dir, "requirements.txt")

    discordia_archivum_url = "https://github.com/Linaqruf/discordia-archivum"
    discordia_archivum_dir = os.path.join(repositories_dir, "discordia-archivum")
    discordia_archivum_deps = os.path.join(discordia_archivum_dir, "requirements.txt")

    clone_repo(infinite_image_browser_url, infinite_image_browser_dir, "main")
    clone_repo(discordia_archivum_url, discordia_archivum_dir, "main")
    print(infinite_image_browser_deps,"+=+++=====+++=====+++=")
    os.system(f"pip -q install --upgrade -r {infinite_image_browser_deps}")
    os.system("pip -q install python-dotenv")
    os.system(f"pip -q install --upgrade -r {discordia_archivum_deps}")

def install_dependencies():
    requirements_file = os.path.join(repo_dir, "requirements.txt")
    model_util        = os.path.join(repo_dir, "library/model_util.py")
    gpu_info          = getoutput('nvidia-smi')
    t4_xformers_wheel = "https://github.com/Linaqruf/colab-xformers/releases/download/0.0.20/xformers-0.0.20+1d635e1.d20230519-cp310-cp310-linux_x86_64.whl"

    os.system("apt install aria2 lz4 -y")
    os.system("wget -c https://github.com/camenduru/gperftools/releases/download/v1.0/libtcmalloc_minimal.so.4 -O /root/content/libtcmalloc_minimal.so.4")
    os.system(f"pip -q install --upgrade -r {requirements_file}")

    if '2.0.1+cu118' in torch.__version__:
        if 'T4' in gpu_info:
            os.system(f"pip -q install {t4_xformers_wheel}")
        else:
            os.system("pip -q install xformers==0.0.20")
    else:
        os.system("pip -q install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U")
        os.system("pip -q install xformers==0.0.19 triton==2.0.0 -U")

    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)

def prepare_environment():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

    cuda_path = "/usr/local/cuda-11.8/targets/x86_64-linux/lib/"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{ld_library_path}:{cuda_path}"
    # os.environ["LD_PRELOAD"] = "/root/content/libtcmalloc_minimal.so.4"

def installation():
    os.chdir(root_dir)
    clone_repo(repo_url, repo_dir, branch)
    os.chdir(repo_dir)
    setup_directories()
    install_repository()
    install_dependencies()
    prepare_environment()

os.chdir('/root/')
if len(sys.argv) != 4:
    print("Usage: python kohya_lora_script_V2.py <order> <name> <breed>")
    sys.exit(1)

order_number = sys.argv[1]
animal_type = sys.argv[2]
breed = sys.argv[3]
installation()

os.chdir(root_dir)


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

# Commented out IPython magic to ensure Python compatibility.
# @title ## **1.3. Directory Config**
# @markdown Specify the location of your training data in the following cell. A folder with the same name as your input will be created.
import os

# %store -r

train_data_dir = f"/root/content/LoRA/train_data/{breed}/"  # @param {'type' : 'string'}
# %store train_data_dir

os.makedirs(train_data_dir, exist_ok=True)
print(f"Your train data directory : {train_data_dir}")

os.system("pip install portpicker")

os.chdir('/root/')
if len(sys.argv) != 4:
    print("Usage: python kohya_lora_script_V2.py <order> <name> <breed>")
    sys.exit(1)

order_number = sys.argv[1]
animal_type = sys.argv[2]
breed = sys.argv[3]

# order_number = options.order[0] 
# animal_type = animal_type
order_date = "2023-01-20"
drive_connection = DriveConnection(order_number)

DID_MANUALLY_UPLOAD = True

drive_connection.download_files(folder_path=train_data_dir, folder_id="1lNkSscwxEfq5WcgnSX4o53qjlZqTbc0v")

"""# **III. Data Preprocessing**"""

# Commented out IPython magic to ensure Python compatibility.
# @title ## **3.1. Data Cleaning**
import os
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

# %store -r

os.chdir(root_dir)

test = os.listdir(train_data_dir)
#@markdown This section removes unsupported media types such as `.mp4`, `.webm`, and `.gif`, as well as any unnecessary files.
#@markdown To convert a transparent dataset with an alpha channel (RGBA) to RGB and give it a white background, set the `convert` parameter to `True`.
convert = False  # @param {type:"boolean"}
#@markdown Alternatively, you can give the background a `random_color` instead of white by checking the corresponding option.
random_color = False  # @param {type:"boolean"}
recursive = False

batch_size = 32
supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".npz",
    ".txt",
    ".json",
]

background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

def clean_directory(directory):
    for item in os.listdir(directory):
        file_path = os.path.join(directory, item)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(item)[1]
            if file_ext not in supported_types:
                print(f"Deleting file {item} from {directory}")
                os.remove(file_path)
        elif os.path.isdir(file_path) and recursive:
            clean_directory(file_path)

def process_image(image_path):
    img = Image.open(image_path)
    img_dir, image_name = os.path.split(image_path)

    if img.mode in ("RGBA", "LA"):
        if random_color:
            background_color = random.choice(background_colors)
        else:
            background_color = (255, 255, 255)
        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[-1])

        if image_name.endswith(".webp"):
            bg = bg.convert("RGB")
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            bg.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            bg.save(image_path, "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp"):
            new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
            img.save(new_image_path, "JPEG")
            os.remove(image_path)
            print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
        else:
            img.save(image_path, "PNG")

def find_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".webp"):
                images.append(os.path.join(root, file))
    return images

clean_directory(train_data_dir)
images = find_images(train_data_dir)
num_batches = len(images) // batch_size + 1

if convert:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size
            batch = images[start:end]
            executor.map(process_image, batch)

    print("All images have been converted")

finetune_dir = "/root/content/kohya-trainer/"
os.environ['PYTHONPATH'] = finetune_dir
sys.path.append(finetune_dir)

os.chdir(finetune_dir)

beam_search = True #@param {type:'boolean'}
min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

config = {
    "_train_data_dir"   : train_data_dir,
    "batch_size"        : 8,
    "beam_search"       : beam_search,
    "min_length"        : min_length,
    "max_length"        : max_length,
    "debug"             : True,
    "caption_extension" : ".caption",
    "max_data_loader_n_workers" : 2,
    "recursive"         : True
}

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

os.chdir(finetune_dir)
os.system(f"python finetune/make_captions.py {args}")


#@markdown [Waifu Diffusion 1.4 Tagger V2](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) is a Danbooru-styled image classification model developed by SmilingWolf. It can also be useful for general image tagging, for example, `1girl, solo, looking_at_viewer, short_hair, bangs, simple_background`.
model = "SmilingWolf/wd-v1-4-moat-tagger-v2" #@param ["SmilingWolf/wd-v1-4-moat-tagger-v2", "SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
#@markdown Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.
undesired_tags = "" #@param {type:'string'}
#@markdown Adjust `general_threshold` for pruning tags (less tags, less flexible). `character_threshold` is useful if you want to train with character tags, e.g. `hakurei reimu`.
general_threshold = 0.35 #@param {type:"slider", min:0, max:1, step:0.05}
character_threshold = 0.85 #@param {type:"slider", min:0, max:1, step:0.05}

config = {
    "_train_data_dir"           : train_data_dir,
    "batch_size"                : 8,
    "repo_id"                   : model,
    "recursive"                 : True,
    "remove_underscore"         : True,
    "general_threshold"         : general_threshold,
    "character_threshold"       : character_threshold,
    "caption_extension"         : ".txt",
    "max_data_loader_n_workers" : 2,
    "debug"                     : True,
    "undesired_tags"            : undesired_tags
}

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

os.system(f"python finetune/tag_images_by_wd14_tagger.py {args}")

root_dir = '/root/content/'
os.chdir(root_dir)

# @markdown Add or remove custom tags here.
extension   = ".txt"  # @param [".txt", ".caption"]
custom_tag  = "Animal"  # @param {type:"string"}
# @markdown Use `sub_folder` option to specify a subfolder for multi-concept training.
# @markdown > Specify `--all` to process all subfolders/`recursive`
sub_folder  = "" #@param {type: "string"}
# @markdown Enable this to append custom tags at the end of lines.
append      = False  # @param {type:"boolean"}
# @markdown Enable this if you want to remove captions/tags instead.
remove_tag  = False  # @param {type:"boolean"}
recursive   = False

if sub_folder == "":
    image_dir = train_data_dir
elif sub_folder == "--all":
    image_dir = train_data_dir
    recursive = True
elif sub_folder.startswith("/content"):
    image_dir = sub_folder
else:
    image_dir = os.path.join(train_data_dir, sub_folder)
    os.makedirs(image_dir, exist_ok=True)

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def process_tags(filename, custom_tag, append, remove_tag):
    contents = read_file(filename)
    tags = [tag.strip() for tag in contents.split(',')]
    custom_tags = [tag.strip() for tag in custom_tag.split(',')]

    for custom_tag in custom_tags:
        custom_tag = custom_tag.replace("_", " ")
        if remove_tag:
            while custom_tag in tags:
                tags.remove(custom_tag)
        else:
            if custom_tag not in tags:
                if append:
                    tags.append(custom_tag)
                else:
                    tags.insert(0, custom_tag)

    contents = ', '.join(tags)
    write_file(filename, contents)

def process_directory(image_dir, tag, append, remove_tag, recursive):
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)

        if os.path.isdir(file_path) and recursive:
            process_directory(file_path, tag, append, remove_tag, recursive)
        elif filename.endswith(extension):
            process_tags(file_path, tag, append, remove_tag)

tag = custom_tag

if not any(
    [filename.endswith(extension) for filename in os.listdir(image_dir)]
):
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            open(
                os.path.join(image_dir, filename.split(".")[0] + extension),
                "w",
            ).close()

if custom_tag:
    process_directory(image_dir, tag, append, remove_tag, recursive)


# Commented out IPython magic to ensure Python compatibility.
# @title ## **3.4. Bucketing and Latents Caching**
# %store -r
import time
# @markdown This code will create buckets based on the `bucket_resolution` provided for multi-aspect ratio training, and then convert all images within the `train_data_dir` to latents.
bucketing_json    = os.path.join(training_dir, "meta_lat.json")
metadata_json     = os.path.join(training_dir, "meta_clean.json")
bucket_resolution = 1024  # @param {type:"slider", min:512, max:1024, step:128}
mixed_precision   = "no"  # @param ["no", "fp16", "bf16"] {allow-input: false}
skip_existing     = False  # @param{type:"boolean"}
flip_aug          = False  # @param{type:"boolean"}
# @markdown Use `clean_caption` option to clean such as duplicate tags, `women` to `girl`, etc
clean_caption     = False #@param {type:"boolean"}
#@markdown Use the `recursivcontent/training_images/e` option to process subfolders as well
recursive         = True #@param {type:"boolean"}

metadata_config = {
    "_train_data_dir": train_data_dir,
    "_out_json": metadata_json,
    "recursive": recursive,
    "full_path": recursive,
    "clean_caption": clean_caption
}

bucketing_config = {
    "_train_data_dir": train_data_dir,
    "_in_json": metadata_json,
    "_out_json": bucketing_json,
    "_model_name_or_path": vae_path if vae_path else model_path,
    "recursive": recursive,
    "full_path": recursive,
    "flip_aug": flip_aug,
    "skip_existing": skip_existing,
    "batch_size": 4,
    "max_data_loader_n_workers": 2,
    "max_resolution": f"{bucket_resolution}, {bucket_resolution}",
    "mixed_precision": mixed_precision,
}

def generate_args(config):
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
    return args.strip()

merge_metadata_args = generate_args(metadata_config)
prepare_buckets_args = generate_args(bucketing_config)

os.chdir(finetune_dir)
os.system(f"python finetune/merge_all_to_metadata.py {merge_metadata_args}")
time.sleep(1)
os.system(f"python finetune/prepare_buckets_latents.py {prepare_buckets_args}")

"""# **IV. Training**


"""

import toml

# @title ## **4.1. LoRa: Low-Rank Adaptation Config**
# @markdown Kohya's `LoRA` renamed to `LoRA-LierLa` and Kohya's `LoCon` renamed to `LoRA-C3Lier`, read [official announcement](https://github.com/kohya-ss/sd-scripts/blob/849bc24d205a35fbe1b2a4063edd7172533c1c01/README.md#naming-of-lora).
network_category = "LoRA_LierLa"  # @param ["LoRA_LierLa", "LoRA_C3Lier", "DyLoRA_LierLa", "DyLoRA_C3Lier", "LoCon", "LoHa", "IA3", "LoKR", "DyLoRA_Lycoris"]

# @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha | unit |
# @markdown | :---: | :---: | :---: | :---: | :---: | :---: |
# @markdown | LoRA-LierLa | 32 | 1 | - | - | - |
# @markdown | LoCon/LoRA-C3Lier | 16 | 8 | 8 | 1 | - |
# @markdown | LoHa | 8 | 4 | 4 | 1 | - |
# @markdown | Other Category | ? | ? | ? | ? | - |

# @markdown Specify `network_args` to add `optional` training args, like for specifying each 25 block weight, read [this](https://github.com/kohya-ss/sd-scripts/blob/main/train_network_README-ja.md#%E9%9A%8E%E5%B1%A4%E5%88%A5%E5%AD%A6%E7%BF%92%E7%8E%87)
network_args    = ""  # @param {'type':'string'}

# @markdown ### **Linear Layer Config**
# @markdown Used by all `network_category`. When in doubt, set `network_dim = network_alpha`
network_dim     = 32  # @param {'type':'number'}
network_alpha   = 16  # @param {'type':'number'}

# @markdown ### **Convolutional Layer Config**
# @markdown Only required if `network_category` is not `LoRA_LierLa`, as it involves training convolutional layers in addition to linear layers.
conv_dim        = 32  # @param {'type':'number'}
conv_alpha      = 16  # @param {'type':'number'}

# @markdown ### **DyLoRA Config**
# @markdown Only required if `network_category` is `DyLoRA_LierLa` and `DyLoRA_C3Lier`
unit = 4  # @param {'type':'number'}

if isinstance(network_args, str):
    network_args = network_args.strip()
    if network_args.startswith('[') and network_args.endswith(']'):
        try:
            network_args = ast.literal_eval(network_args)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing network_args: {e}\n")
            network_args = []
    elif len(network_args) > 0:
        print(f"WARNING! '{network_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
        network_args = []
    else:
        network_args = []
else:
    network_args = []

network_config = {
    "LoRA_LierLa": {
        "module": "networks.lora",
        "args"  : []
    },
    "LoRA_C3Lier": {
        "module": "networks.lora",
        "args"  : [
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "DyLoRA_LierLa": {
        "module": "networks.dylora",
        "args"  : [
            f"unit={unit}"
        ]
    },
    "DyLoRA_C3Lier": {
        "module": "networks.dylora",
        "args"  : [
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}",
            f"unit={unit}"
        ]
    },
    "LoCon": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=locon",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "LoHa": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=loha",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "IA3": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=ia3",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "LoKR": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=lokr",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    },
    "DyLoRA_Lycoris": {
        "module": "lycoris.kohya",
        "args"  : [
            f"algo=dylora",
            f"conv_dim={conv_dim}",
            f"conv_alpha={conv_alpha}"
        ]
    }
}

network_module = network_config[network_category]["module"]
network_args.extend(network_config[network_category]["args"])

lora_config = {
    "additional_network_arguments": {
        "no_metadata"                     : False,
        "network_module"                  : network_module,
        "network_dim"                     : network_dim,
        "network_alpha"                   : network_alpha,
        "network_args"                    : network_args,
        "network_train_unet_only"         : True,
        "training_comment"                : None,
    },
}

print(toml.dumps(lora_config))


# @title ## **4.2. Optimizer Config**
# @markdown Use `Adafactor` optimizer. `RMSprop 8bit` or `Adagrad 8bit` may work. `AdamW 8bit` doesn't seem to work.
optimizer_type = "AdaFactor"  # @param ["AdamW", "AdamW8bit", "Lion8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation(DAdaptAdamPreprint)", "DAdaptAdaGrad", "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor"]
# @markdown Specify `optimizer_args` to add `additional` args for optimizer, e.g: `["weight_decay=0.6"]`
optimizer_args = "[ \"scale_parameter=False\", \"relative_step=False\", \"warmup_init=False\" ]"  # @param {'type':'string'}
# @markdown ### **Learning Rate Config**
# @markdown Different `optimizer_type` and `network_category` for some condition requires different learning rate. It's recommended to set `text_encoder_lr = 1/2 * unet_lr`
learning_rate = 1e-4  # @param {'type':'number'}
# @markdown ### **LR Scheduler Config**
# @markdown `lr_scheduler` provides several methods to adjust the learning rate based on the number of epochs.
lr_scheduler = "constant_with_warmup"  # @param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
lr_warmup_steps = 100  # @param {'type':'number'}
# @markdown Specify `lr_scheduler_num` with `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial`
lr_scheduler_num = 0  # @param {'type':'number'}

if isinstance(optimizer_args, str):
    optimizer_args = optimizer_args.strip()
    if optimizer_args.startswith('[') and optimizer_args.endswith(']'):
        try:
            optimizer_args = ast.literal_eval(optimizer_args)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing optimizer_args: {e}\n")
            optimizer_args = []
    elif len(optimizer_args) > 0:
        print(f"WARNING! '{optimizer_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
        optimizer_args = []
    else:
        optimizer_args = []
else:
    optimizer_args = []

optimizer_config = {
    "optimizer_arguments": {
        "optimizer_type"          : optimizer_type,
        "learning_rate"           : learning_rate,
        "max_grad_norm"           : 0,
        "optimizer_args"          : optimizer_args,
        "lr_scheduler"            : lr_scheduler,
        "lr_warmup_steps"         : lr_warmup_steps,
        "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
        "lr_scheduler_type"       : None,
        "lr_scheduler_args"       : None,
    },
}

print(toml.dumps(optimizer_config))


# @title ## **4.3. Advanced Training Config** (Optional)
import toml


# @markdown ### **Optimizer State Config**
save_optimizer_state      = False #@param {type:"boolean"}
load_optimizer_state      = "" #@param {type:"string"}
# @markdown ### **Noise Control**
noise_control_type        = "none" #@param ["none", "noise_offset", "multires_noise"]
# @markdown #### **a. Noise Offset**
# @markdown Control and easily generating darker or light images by offset the noise when fine-tuning the model. Recommended value: `0.1`. Read [Diffusion With Offset Noise](https://www.crosslabs.org//blog/diffusion-with-offset-noise)
noise_offset_num          = 0.0357  # @param {type:"number"}
# @markdown **[Experimental]**
# @markdown Automatically adjusts the noise offset based on the absolute mean values of each channel in the latents when used with `--noise_offset`. Specify a value around 1/10 to the same magnitude as the `--noise_offset` for best results. Set `0` to disable.
adaptive_noise_scale      = 0.00357 # @param {type:"number"}
# @markdown #### **b. Multires Noise**
# @markdown enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)
multires_noise_iterations = 6 #@param {type:"slider", min:1, max:10, step:1}
multires_noise_discount = 0.3 #@param {type:"slider", min:0.1, max:1, step:0.1}
# @markdown ### **Caption Dropout**
caption_dropout_rate = 0  # @param {type:"number"}
caption_tag_dropout_rate = 0.5  # @param {type:"number"}
caption_dropout_every_n_epochs = 0  # @param {type:"number"}
# @markdown ### **Custom Train Function**
# @markdown Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends `5`. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma             = 5 #@param {type:"number"}

advanced_training_config = {
    "advanced_training_config": {
        "resume"                        : load_optimizer_state,
        "save_state"                    : save_optimizer_state,
        "save_last_n_epochs_state"      : save_optimizer_state,
        "noise_offset"                  : noise_offset_num if noise_control_type == "noise_offset" else None,
        "adaptive_noise_scale"          : adaptive_noise_scale if adaptive_noise_scale and noise_control_type == "noise_offset" else None,
        "multires_noise_iterations"     : multires_noise_iterations if noise_control_type =="multires_noise" else None,
        "multires_noise_discount"       : multires_noise_discount if noise_control_type =="multires_noise" else None,
        "caption_dropout_rate"          : caption_dropout_rate,
        "caption_tag_dropout_rate"      : caption_tag_dropout_rate,
        "caption_dropout_every_n_epochs": caption_dropout_every_n_epochs,
        "min_snr_gamma"                 : min_snr_gamma if not min_snr_gamma == -1 else None,
    }
}

print(toml.dumps(advanced_training_config))

# Commented out IPython magic to ensure Python compatibility.
# @title ## **4.4. Training Config**

# @markdown ### **Project Config**
project_name                = "sdxl_lora"  # @param {type:"string"}
# @markdown Get your `wandb_api_key` [here](https://wandb.ai/settings) to logs with wandb.
wandb_api_key               = "" # @param {type:"string"}
in_json                     = "/root/content/LoRA/meta_lat.json"  # @param {type:"string"}
# @markdown ### **SDXL Config**
gradient_checkpointing      = True  # @param {type:"boolean"}
no_half_vae                 = True  # @param {type:"boolean"}
#@markdown Recommended parameter for SDXL training but if you enable it, `shuffle_caption` won't work
cache_text_encoder_outputs  = False  # @param {type:"boolean"}
#@markdown These options can be used to train U-Net with different timesteps. The default values are 0 and 1000.
min_timestep                = 0 # @param {type:"number"}
max_timestep                = 1000 # @param {type:"number"}
# @markdown ### **Dataset Config**
num_repeats                 = 1  # @param {type:"number"}
resolution                  = 1024  # @param {type:"slider", min:512, max:1024, step:128}
keep_tokens                 = 0  # @param {type:"number"}
# @markdown ### **General Config**
num_epochs                  = 10  # @param {type:"number"}
train_batch_size            = 4  # @param {type:"number"}
mixed_precision             = "fp16"  # @param ["no","fp16","bf16"] {allow-input: false}
seed                        = -1  # @param {type:"number"}
optimization                = "scaled dot-product attention" # @param ["xformers", "scaled dot-product attention"]
# @markdown ### **Save Output Config**
save_precision              = "fp16"  # @param ["float", "fp16", "bf16"] {allow-input: false}
save_every_n_epochs         = 1  # @param {type:"number"}
# @markdown ### **Sample Prompt Config**
enable_sample               = True  # @param {type:"boolean"}
sampler                     = "euler_a"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
positive_prompt             = ""
negative_prompt             = ""
quality_prompt              = "NovelAI"  # @param ["None", "Waifu Diffusion 1.5", "NovelAI", "AbyssOrangeMix", "Stable Diffusion XL"] {allow-input: false}
if quality_prompt          == "NovelAI":
    positive_prompt         = "masterpiece, best quality, "
    negative_prompt         = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, "
if quality_prompt          == "AbyssOrangeMix":
    positive_prompt         = "masterpiece, best quality, "
    negative_prompt         = "(worst quality, low quality:1.4), "
if quality_prompt          == "Stable Diffusion XL":
    negative_prompt         = "3d render, smooth, plastic, blurry, grainy, low-resolution, deep-fried, oversaturated"
custom_prompt               = "face focus, cute, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck" # @param {type:"string"}
# @markdown Specify `prompt_from_caption` if you want to use caption as prompt instead. Will be chosen randomly.
prompt_from_caption         = "none"  # @param ["none", ".txt", ".caption"]
if prompt_from_caption     != "none":
    custom_prompt           = ""
num_prompt                  = 2  # @param {type:"number"}
logging_dir                 = os.path.join(training_dir, "logs")
lowram                      = int(next(line.split()[1] for line in open('/proc/meminfo') if "MemTotal" in line)) / (1024**2) < 15

os.chdir(repo_dir)

prompt_config = {
    "prompt": {
        "negative_prompt" : negative_prompt,
        "width"           : resolution,
        "height"          : resolution,
        "scale"           : 12,
        "sample_steps"    : 28,
        "subset"          : [],
    }
}

train_config = {
    "sdxl_arguments": {
        "cache_text_encoder_outputs" : cache_text_encoder_outputs,
        "no_half_vae"                : True,
        "min_timestep"               : min_timestep,
        "max_timestep"               : max_timestep,
        "shuffle_caption"            : True if not cache_text_encoder_outputs else False,
        "lowram"                     : lowram
    },
    "model_arguments": {
        "pretrained_model_name_or_path" : model_path,
        "vae"                           : vae_path,
    },
    "dataset_arguments": {
        "debug_dataset"                 : False,
        "in_json"                       : in_json,
        "train_data_dir"                : train_data_dir,
        "dataset_repeats"               : num_repeats,
        "keep_tokens"                   : keep_tokens,
        "resolution"                    : str(resolution) + ',' + str(resolution),
        "color_aug"                     : False,
        "face_crop_aug_range"           : None,
        "token_warmup_min"              : 1,
        "token_warmup_step"             : 0,
    },
    "training_arguments": {
        "output_dir"                    : os.path.join(output_dir, project_name),
        "output_name"                   : project_name if project_name else "last",
        "save_precision"                : save_precision,
        "save_every_n_epochs"           : save_every_n_epochs,
        "save_n_epoch_ratio"            : None,
        "save_last_n_epochs"            : None,
        "resume"                        : None,
        "train_batch_size"              : train_batch_size,
        "max_token_length"              : 225,
        "mem_eff_attn"                  : False,
        "sdpa"                          : True if optimization == "scaled dot-product attention" else False,
        "xformers"                      : True if optimization == "xformers" else False,
        "max_train_epochs"              : num_epochs,
        "max_data_loader_n_workers"     : 8,
        "persistent_data_loader_workers": True,
        "seed"                          : seed if seed > 0 else None,
        "gradient_checkpointing"        : gradient_checkpointing,
        "gradient_accumulation_steps"   : 1,
        "mixed_precision"               : mixed_precision,
    },
    "logging_arguments": {
        "log_with"          : "wandb" if wandb_api_key else "tensorboard",
        "log_tracker_name"  : project_name if wandb_api_key and not project_name == "last" else None,
        "logging_dir"       : logging_dir,
        "log_prefix"        : project_name if not wandb_api_key else None,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps"    : None,
        "sample_every_n_epochs"   : save_every_n_epochs if enable_sample else None,
        "sample_sampler"          : sampler,
    },
    "saving_arguments": {
        "save_model_as": "safetensors"
    },
}

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

def prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt):
    if enable_sample:
        search_pattern = os.path.join(train_data_dir, '**/*' + prompt_from_caption)
        caption_files = glob.glob(search_pattern, recursive=True)

        if not caption_files:
            if not custom_prompt:
                custom_prompt = "masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"
            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = [
                {"prompt": positive_prompt + custom_prompt if positive_prompt else custom_prompt}
            ]
        else:
            selected_files = random.sample(caption_files, min(num_prompt, len(caption_files)))

            prompts = []
            for file in selected_files:
                with open(file, 'r') as f:
                    prompts.append(f.read().strip())

            new_prompt_config = prompt_config.copy()
            new_prompt_config['prompt']['subset'] = []

            for prompt in prompts:
                new_prompt = {
                    "prompt": positive_prompt + prompt if positive_prompt else prompt,
                }
                new_prompt_config['prompt']['subset'].append(new_prompt)

        return new_prompt_config
    else:
        return prompt_config

def eliminate_none_variable(config):
    for key in config:
        if isinstance(config[key], dict):
            for sub_key in config[key]:
                if config[key][sub_key] == "":
                    config[key][sub_key] = None
        elif config[key] == "":
            config[key] = None

    return config

try:
    train_config.update(optimizer_config)
except NameError:
    raise NameError("'optimizer_config' dictionary is missing. Please run  '4.1. Optimizer Config' cell.")

try:
    train_config.update(lora_config)
except NameError:
    raise NameError("'lora_config' dictionary is missing. Please run  '4.1. LoRa: Low-Rank Adaptation Config' cell.")

advanced_training_warning = False
try:
    train_config.update(advanced_training_config)
except NameError:
    advanced_training_warning = True
    pass

prompt_config = prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompt)

config_path         = os.path.join(config_dir, "config_file.toml")
prompt_path         = os.path.join(config_dir, "sample_prompt.toml")

config_str          = toml.dumps(eliminate_none_variable(train_config))
prompt_str          = toml.dumps(eliminate_none_variable(prompt_config))

write_file(config_path, config_str)
write_file(prompt_path, prompt_str)

print(config_str)

if advanced_training_warning:
    import textwrap
    error_message = "WARNING: This is not an error message, but the [advanced_training_config] dictionary is missing. Please run the '4.2. Advanced Training Config' cell if you intend to use it, or continue to the next step."
    wrapped_message = textwrap.fill(error_message, width=80)
    print('\033[38;2;204;102;102m' + wrapped_message + '\033[0m\n')
    pass

print(prompt_str)

#@title ## **4.5. Start Training**

#@markdown Check your config here if you want to edit something:
#@markdown - `sample_prompt` : /content/LoRA/config/sample_prompt.toml
#@markdown - `config_file` : /content/LoRA/config/config_file.toml


#@markdown You can import config from another session if you want.

sample_prompt   = "/root/content/LoRA/config/sample_prompt.toml" #@param {type:'string'}
config_file     = "/root/content/LoRA/config/config_file.toml" #@param {type:'string'}

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents

def train(config):
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

    return args

accelerate_conf = {
    "config_file" : "/root/content/kohya-trainer/accelerate_config/config.yaml",
    "num_cpu_threads_per_process" : 1,
}

train_conf = {
    "sample_prompts"  : sample_prompt if os.path.exists(sample_prompt) else None,
    "config_file"     : config_file,
    "wandb_api_key"   : wandb_api_key if wandb_api_key else None
}

accelerate_args = train(accelerate_conf)
train_args = train(train_conf)

os.chdir(repo_dir)
os.system(f"accelerate launch {accelerate_args} sdxl_train_network.py {train_args}")