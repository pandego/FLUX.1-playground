import asyncio
import aiohttp
import os
import runpod
from runpod import AsyncioEndpoint, AsyncioJob
from dotenv import load_dotenv
from datetime import datetime
import json
import base64
from loguru import logger
from rich import print as rprint
import argparse

parser = argparse.ArgumentParser(description="Run a job on Runpod.")
parser.add_argument("--input", "-i", type=str, help="Path to the input JSON file.")
parser.add_argument(
    "--output", type=str, default="output", help="Path to the output PNG file."
)

load_dotenv(".env", verbose=True, override=True)

# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # For Windows users.

runpod.api_key = os.getenv("RUNPOD_API_KEY")
enpoint_id = os.getenv("ENDPOINT_ID")


# ------------------------- #
# --- Utility functions --- #
# ------------------------- #

# TODO: Move utility functions to a separate file


def load_json_file(json_file_path):
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)
            logger.success("Successfully loaded JSON file.")
            return data
    except FileNotFoundError:
        logger.error(
            "File not found. Please ensure the JSON file exists in the specified path."
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def decode_and_save_image(json_input, output_image_path):
    try:
        # Determine if the input is a file path or a JSON object
        if isinstance(json_input, str):
            with open(json_input, "r") as file:
                data = json.load(file)
                logger.success("Successfully loaded JSON file.")
        elif isinstance(json_input, dict):
            data = json_input
            logger.success("Successfully received JSON object.")
        else:
            logger.error(
                "Invalid input: json_input must be a file path or a JSON object."
            )
            raise ValueError(
                "Invalid input: json_input must be a file path or a JSON object."
            )

        # Extracting the base64 encoded image data
        base64_image = data.get("image")

        # Decode the Base64 string
        decoded_image_data = base64.b64decode(base64_image)

        # Writing the decoded data to an image file
        with open(output_image_path, "wb") as image_file:
            image_file.write(decoded_image_data)

        logger.success(
            f"Image successfully decoded and saved as '{output_image_path}'."
        )

    except FileNotFoundError:
        logger.error(
            "File not found. Please ensure the JSON file exists in the specified path."
        )
    except KeyError as e:
        logger.error(f"Error in JSON structure: {e}")
    except ValueError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


# --------------------- #
# --- Main function --- #
# --------------------- #


async def main(input_payload):
    output = None  # Initialize output to None
    async with aiohttp.ClientSession() as session:

        endpoint = AsyncioEndpoint(enpoint_id, session)
        job: AsyncioJob = await endpoint.run(input_payload)

        start_time = datetime.now()

        # Polling job status
        while True:
            status = await job.status()
            logger.info(f"Current job status: {status}")
            if status == "COMPLETED":
                output = await job.output()
                logger.success("Job completed successfully!")
                break  # Exit the loop once the job is completed.
            elif status in ["FAILED"]:
                logger.error("Job failed or encountered an error.")
                break
            else:
                logger.info("Job in queue or processing. Waiting 6 seconds...")
                await asyncio.sleep(6)  # Wait for 6 seconds before polling again

        end_time = datetime.now()
        logger.info(f"Time taken: {end_time - start_time}")

    return output


if __name__ == "__main__":

    args = parser.parse_args()

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    if args.input:
        input_payload = load_json_file(args.input).get("input")
    else:
        input_payload = {
            "prompt": "An astronaut floating in a jungle, cold color palette, muted colors, detailed, 8k",
            "model_name": "black-forest-labs/FLUX.1-dev",
            "width": 1024,
            "height": 1024,
            "guidance_scale": 3.5,
            "seed": 42,
        }

    rprint(f"Input payload:\n{input_payload}\n")

    json_output = asyncio.run(main(input_payload))

    # rprint(json_output)

    date_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_folder}/{datetime.now().strftime('%Y%m%d_%HH%MM%SS')}_output_flux.{input_payload.get('model_name').split('-')[-1]}_seed_{input_payload.get('seed')}.png"

    # Convert output JSON response to PNG image
    decode_and_save_image(json_output, output_path)
