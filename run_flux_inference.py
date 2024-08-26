import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
import os
from tqdm import tqdm
from loguru import logger


# TODO: Clean this into a nice script
# TODO: Add the following arguments to the script
# use_lora = True
# little_vram = True
# device_generation = "cuda"
# flux_version = "1-dev"
# lorapath = "pandego/flux.1-dev_LoRa_AC"
# prompts = ""
# height=1024
# width=1280
# guidance_scale=3.5

# ---------------------- #
# --- Load Variables --- #
# ---------------------- #

load_dotenv(".env", verbose=True, override=True)
logger.success(f"HF_TOKEN has been set: '{os.environ['HF_TOKEN'][:3] + '**************'}'")

seed: int = 123
little_vram: bool = True
device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
flux_version: str = "1-dev"  # 1-schnell or 1-dev
num_inference_steps: int = 4 if flux_version == "1-schnell" else 50
use_lora: bool = False
if use_lora:
    logger.info(f"Using LoRa")
    lora_path = "pandego/flux.1-dev_LoRa"

prompts: list | str = [
     "A professional photo of a man that looks like [trigger], warm smile, looking at the camera, with a office blurry background",
]

prompts = "A professional photo of a man that looks like [trigger], warm smile, looking at the camera, with a office blurry background"


# --------------------------- #
# --- Load the Base Model --- #
# --------------------------- #

base_model = f"black-forest-labs/FLUX.{flux_version}"

pipe = FluxPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16
)

pipe.to(device)

if little_vram:
    pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# --------------------- #
# --- Load the LoRa --- #
# --------------------- #

if use_lora:
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=lora_path,
    )

# ---------------------- #
# --- Generate Image --- #
# ---------------------- #



if isinstance(prompts, list):
    i = 0
    for prompt in tqdm(prompts):
        output = pipe(
            prompt,
            output_type="pil",  # optional
            height=1024,  # optional
            width=1280,  # optional
            guidance_scale=3.5,  # optional
            max_sequence_length=512,  # optional
            num_inference_steps=num_inference_steps,  # use a larger number if you are using [dev]
            generator=torch.Generator(device).manual_seed(seed),
        ).images[0]

        output.save(f"prompt_{i}_flux.{flux_version}.png")
        i += 1

elif isinstance(prompts, str):
    output = pipe(
        prompts,
        output_type="pil",  # optional
        height=1024,  # optional
        width=1280,  # optional
        guidance_scale=3.5,  # optional
        max_sequence_length=512,  # optional
        num_inference_steps=num_inference_steps,  # use a larger number if you are using [dev]
        generator=torch.Generator(device).manual_seed(seed)
    ).images[0]

    output.save(f"prompt_flux.{flux_version}.png")