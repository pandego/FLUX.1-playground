import argparse
import os
import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
from loguru import logger
from transformers import BitsAndBytesConfig, QuantoConfig
from datetime import datetime

# TODO: Move all arguments to a json file

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flux Image Generation Script")
    parser.add_argument(
        "--little_vram", type=bool, default=False, help="Enable little VRAM mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for generation (cuda, mps, cpu)",
    )
    parser.add_argument("--multi_gpu", type=bool, default=True, help="Enable multi-GPU")
    parser.add_argument(
        "--dtype",
        type=torch.dtype,
        default=torch.float16,
        help="Data type to use for generation (torch.float16, torch.bfloat16)",
    )
    parser.add_argument(
        "--flux_version",
        type=str,
        default="1-dev",
        help="Flux version, either '1-dev' or '1-schnell'.",
    )
    parser.add_argument(
        "--lorapath",
        type=str,
        # default="XLabs-AI/flux-lora-collection",  # it should be optional
        help="Path to LoRa weights",
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        # default="realism_lora.safetensors",  # it should be optional
        help="Path to LoRa weights",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="""
        A minimalist and stylized illustration of a cute, fluffy cool looking white dog wearing round, reflective sunglasses.
        The dog is holding a sign that says "Cool Doguito!", and it has a small, button-like nose. 
        The background is solid black, creating a high contrast that highlights the dog's white fur and the dark lenses of the sunglasses. 
        The overall style is clean, simple, and modern, with a focus on the dog's face and the cool vibe of the sunglasses.
        """,
        help="Prompt for image generation",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length, cannot be greater than 512",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument(
        "--guidance_scale", type=float, default=3.5, help="Guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation (for reproducibility)",
    )
    return parser.parse_args()


def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.ipc_collect()


def setup_environment():
    # TODO: catch exception if HF_TOKEN is not set
    load_dotenv(".env", verbose=True, override=True)
    logger.success(
        f"HF_TOKEN has been set: '{os.environ['HF_TOKEN'][:3] + '**************'}'"
    )


def get_device(args):
    # TODO: catch exception if device is not available
    if args.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def print_cuda_devices():
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available.")
        return

    num_devices = torch.cuda.device_count()
    logger.info(f"Number of CUDA devices: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        device_vram = torch.cuda.get_device_properties(i).total_memory / (
                1024 ** 2
        )  # Convert to MB
        logger.info(f"Device {i}: {device_name}, VRAM: {device_vram:.2f} MB")


def load_model(args, device):
    base_model = f"black-forest-labs/FLUX.{args.flux_version}"

    try:
        
        if args.multi_gpu:
            # Normal VRAM mode using multiple GPUs
            pipe = FluxPipeline.from_pretrained(
                base_model,
                low_cpu_mem_usage=True,  # needed if using "balanced" for device_map
                device_map="balanced",
                max_memory={
                    0: "15000MB",
                    1: "15000MB",
                    2: "7000MB"
                },
                torch_dtype=args.dtype,
            )
            # Do not move the pipeline to a single device
        
        else:
            pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=args.dtype)
            pipe.to(device)

        if args.little_vram and args.multi_gpu is False:
            # Little VRAM mode with CPU offloading
            pipe.enable_model_cpu_offload()
            logger.warning("Pipeline offloaded onto the CPU.")
        
        if args.lorapath:
            logger.info(f"Using LoRa from {args.lorapath}")

            pipe.load_lora_weights(
                args.lorapath,
                weight_name=args.weights_name if args.weights_name is not None else None,
                # use_safetensors=True,
                torch_dtype=args.dtype,
                # assign=True
            )
            # pipe.to(device)
        

        return pipe

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_images(pipe, args):
    seed = (
        args.seed if args.seed is not None else torch.randint(0, 2 ** 32 - 1, (1,)).item()
    )
    logger.info(f"Using seed: {seed}")

    num_inference_steps = 4 if args.flux_version == "1-schnell" else 50

    output = pipe(
        args.prompt,
        output_type="pil",
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_path = f"{output_folder}/{datetime.now().strftime('%Y%m%d_%HH%MM%SS')}_output_flux.{args.flux_version}_seed_{seed}.png"
    output.save(output_path)

    logger.success(f"Saved image: '{output_path}'")


def main():
    flush()
    # TODO: add logging and catch exceptions
    args = parse_arguments()
    setup_environment()
    print_cuda_devices()
    device = get_device(args)
    pipe = load_model(args, device)
    generate_images(pipe, args)
    flush()


if __name__ == "__main__":
    # TODO: Add an actual json file as input
    main()
