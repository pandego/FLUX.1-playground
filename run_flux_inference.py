import argparse
import os
import torch
from diffusers import FluxPipeline
from dotenv import load_dotenv
from loguru import logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flux Image Generation Script")
    parser.add_argument("--use_lora", type=bool, default=False, help="Whether to use a LoRa")
    parser.add_argument("--little_vram", type=bool, default=True, help="Enable little VRAM mode")
    parser.add_argument("--device", type=str, default="cuda", help=" Device to use for generation (cuda, mps, cpu)")
    parser.add_argument("--flux_version", type=str, default="1-dev", help="Flux version")
    parser.add_argument("--lorapath", type=str, default="pandego/flux.1-dev_LoRa_AC", help="Path to LoRa weights")
    parser.add_argument("--prompt", type=str, 
                        default="""
                        A minimalist and stylized illustration of a cute, fluffy cool looking white dog wearing round, reflective sunglasses.
                        The dog is holding a sign that says "Cool Doguito!", and it has a small, button-like nose. 
                        The background is solid black, creating a high contrast that highlights the dog's white fur and the dark lenses of the sunglasses. 
                        The overall style is clean, simple, and modern, with a focus on the dog's face and the cool vibe of the sunglasses.
                        """, 
                        help="Prompt for image generation")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Maximum sequence length, cannot be greater than 512")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation (for reproducibility)")
    return parser.parse_args()

def setup_environment():
    # TODO: catch exception if HF_TOKEN is not set
    load_dotenv(".env", verbose=True, override=True)
    logger.success(f"HF_TOKEN has been set: '{os.environ['HF_TOKEN'][:3] + '**************'}'")

def get_device(args):
    # TODO: catch exception if device is not available
    if args.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model(args, device):
    base_model = f"black-forest-labs/FLUX.{args.flux_version}"
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    pipe.to(device)
    
    if args.little_vram:
        pipe.enable_model_cpu_offload()
    
    if args.use_lora:
        logger.info(f"Using LoRa from {args.lorapath}")
        pipe.load_lora_weights(args.lorapath)
    
    return pipe

def generate_images(pipe, args):
    seed = args.seed if args.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
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
        generator=torch.Generator(args.device).manual_seed(seed)
    ).images[0]
    
    output.save(f"output_flux.{args.flux_version}_seed_{seed}.png")
    logger.info(f"Saved image: output_flux.{args.flux_version}_seed_{seed}.png")

def main():
    # TODO: add logging and catch exceptions
    args = parse_arguments()
    setup_environment()
    device = get_device(args)
    pipe = load_model(args, device)
    generate_images(pipe, args)

if __name__ == "__main__":
    main()
    