import runpod
import torch
from diffusers import FluxPipeline
from io import BytesIO
import base64
from PIL import Image
from loguru import logger


def load_model(
    model_name: str,
    lora_model_name: str = None,
    weight_name: str = None,
    # -- Coming soon -- #
    # lora_model_names: list = None,
    # weight_names: list = None
) -> FluxPipeline:
    """
    Load the FluxPipeline model with optional LoRa weights.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    lora_model_name : str, optional
        The name of the LoRa model or collection to load, by default None.
    weight_name : str, optional
        The name of the specific weights to load from the LoRa model, by default None.

    Returns
    -------
    FluxPipeline
        The loaded FluxPipeline model.

    """
    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()  # Save VRAM by offloading the model to CPU

    if lora_model_name:
        logger.info(f"Loading LoRa model or collection from {lora_model_name}")
        if weight_name:
            logger.info(f"Loading weights from {weight_name}")
            adapter_name = weight_name.split(".")[0]
        else:
            adapter_name = lora_model_name.split("/")[-1]

        logger.info(f"Adapter name: {adapter_name}")
        pipe.load_lora_weights(
            lora_model_name, weight_name=weight_name, adapter_name=adapter_name
        )

    return pipe


def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image to a base64 encoded string.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to be converted to base64.

    Returns
    -------
    str
        The base64 encoded string representation of the image.

    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def flux_handler(event):
    global model

    # Get the input parameters from the event
    input_data = event.get("input", {})
    prompt = input_data.get("prompt")
    model_name = input_data.get("model_name", "black-forest-labs/FLUX.1-dev")
    height = input_data.get("height", 1024)
    width = input_data.get("width", 1024)
    # num_inference_steps = input_data.get("num_inference_steps", 50)  # TODO: Implement this
    guidance_scale = input_data.get("guidance_scale", 3.5)
    seed = input_data.get("seed", 42)

    # If using a single LoRa model
    lora_model_name = input_data.get("lora_model_name", None)
    weight_name = input_data.get("weight_name", None)

    # If using multiple LoRa models
    # TODO: Combine multiple adapters to create new and unique images.
    # lora_model_names = input_data.get("lora_model_names", None)
    # weight_names = input_data.get("weight_names", None)

    # Validate input
    if not prompt:
        return {"error": "No prompt provided for image generation."}

    # TODO: Implement caching for the model
    model = load_model(model_name, lora_model_name, weight_name)

    # Get the active adapters or LoRa models
    logger.info(f"Active adapters:\n   {model.get_active_adapters()}")
    logger.info(f"List of adapters:\n   {model.get_list_adapters()}")

    try:
        # Generate the image
        image = model(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=50 if "dev" in model_name else 4,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        # Convert the image to base64
        image_base64 = image_to_base64(image)

        logger.success("Image generated successfully! 🎉")

        return {"image": image_base64, "prompt": prompt}

    except Exception as e:
        logger.error(f"Failed to generate image: {e}")
        return {"error": str(e)}


runpod.serverless.start({"handler": flux_handler})
