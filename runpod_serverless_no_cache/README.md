# FLUX.1-dev Text-to-Image Serverless on RunPod

This project sets up a serverless text-to-image generation service using the FLUX.1-dev model on RunPod. It supports base model inference as well as LoRA (Low-Rank Adaptation) for fine-tuned results.

## Prerequisites

- Docker installed on your local machine
- A RunPod account
- Miniconda or Anaconda installed on your local machine

## Project Structure

```
.
├── Dockerfile
├── environment.yml
├── pyproject.toml
├── handler.py
├── test_input.json
└── README.md
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/pandego/FLUX.1-playground.git
   cd FLUX.1-playground/runpod_serverless_no_cache
   ```

2. Create the conda environment:
   ```
   conda env create -f environment.yml
   ```

3. Activate the conda environment:
   ```
   conda activate runpod-flux
   ```

4. Install project dependencies using poetry:
   ```
   poetry install --no-root
   ```

## Build and Push the Docker Image to the DockerHub

1. Build the Docker image:
   ```
   docker build -t pandego/flux_image_generator_no_cache:latest .
   ```

2. Push the image to DockerHub:
   ```
   docker push pandego/flux_image_generator_no_cache:latest
   ```

## Local Testing

To test the serverless function locally:

1. Ensure you're in the `runpod-flux` conda environment:
   ```
   conda activate runpod-flux
   ```

2. Run the handler locally with the `--rp_server_api` flag (it will use the `test_input.json`):
   ```
   python handler.py --rp_server_api
   ```

## Deploying on RunPod

1. Log in to your RunPod account.

2. Create a new serverless template:
   - Container Image: `pandego/flux_image_generator:latest`
   - Container Disk: Set according to your needs (e.g., 100GB)
   - Select an appropriate GPU (e.g., NVIDIA RTX 4090)

3. Deploy your serverless function using the created template.

4. Use the provided API endpoint to send requests to your serverless function.

## API Usage

Send a POST request to your RunPod endpoint with the following JSON structure:

```json
{
  "input": {
    "prompt": "Your text prompt here",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "seed": 42,
    "model_name": "black-forest-labs/FLUX.1-dev",
    "lora_model_name": "path/to/your/lora/model",
    "weight_name": "optional_weights_name"
  }
}
```

The API will return a JSON response with the generated image in base64 format:

```json
{
  "image": "base64_encoded_image_data",
  "prompt": "Your original prompt"
}
```

## Using the Deployed Endpoint

Once your serverless function is deployed on RunPod, you can interact with it using the RunPod API. Here's how to use your endpoint:

### 1. Start a Job

To start a new job, send a POST request to your endpoint:

```json
curl -X POST https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{
      "input": {
        "prompt": "A cute fluffy white dog in the style of a Pixar animation 3D drawing.",
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "seed": 42,
        "model_name": "black-forest-labs/FLUX.1-dev",
        "lora_model_name": null,
        "weight_name": null
      }
    }'
```

Replace `{YOUR_ENDPOINT_ID}` with your actual endpoint ID and `YOUR_API_KEY` with your RunPod API key.

You'll receive a response with a job ID:

```json
{
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "IN_QUEUE"
}
```

### 2. Check Job Status

Use the job ID to check the status of your job:

```bash
curl https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY'
```

You may see a response indicating the job is still in progress:

```json
{
  "delayTime": 2624,
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "input": {
    "prompt": "A cute fluffy white dog in the style of a Pixar animation 3D drawing."
  },
  "status": "IN_PROGRESS"
}
```

### 3. Get Completed Job Results

Once the job is complete, you'll receive a response like this:

```json
{
  "delayTime": 17158,
  "executionTime": 4633,
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "output": [
    {
      "image": "base64_encoded_image_data",
      "prompt": "A cute fluffy white dog in the style of a Pixar animation 3D drawing."
    }
  ],
  "status": "COMPLETED"
}
```

To save the output:

```bash
curl https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/status/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' | jq . > output.json
```

### 4. Decode and Save the Image

Use the following Python script to decode the base64 image and save it:

```python
import json
import base64

def decode_and_save_image(json_file_path, output_image_path):
    try:
        # Reading the JSON file
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Extracting the base64 encoded image data
        base64_image = data["output"][0]["image"]

        # Decode the Base64 string
        decoded_image_data = base64.b64decode(base64_image)

        # Writing the decoded data to an image file
        with open(output_image_path, "wb") as image_file:
            image_file.write(decoded_image_data)

        print(f"Image successfully decoded and saved as '{output_image_path}'.")

    except FileNotFoundError:
        print("File not found. Please ensure the JSON file exists in the specified path.")
    except KeyError as e:
        print(f"Error in JSON structure: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
json_file_path = "output.json"  # Path to your JSON file
output_image_path = "generated_image.png"  # Desired path for the output image

decode_and_save_image(json_file_path, output_image_path)
```

Run this script to save your generated image.

### Important Notes:

1. Replace `{YOUR_ENDPOINT_ID}` with your actual RunPod endpoint ID in all curl commands.
2. Replace `YOUR_API_KEY` with your actual RunPod API key.
3. You have up to 30 minutes to retrieve your results via the status endpoint for privacy reasons.
4. If you encounter any issues, check the RunPod logs for your serverless function.

By following these steps, you can successfully use your deployed FLUX.1-dev text-to-image generation service on RunPod's serverless platform.

## Notes

- The `lora_model_name` and `weights_name` are optional. If not provided, the base model will be used without LoRA.
- Adjust the `num_inference_steps` in the `handler.py` file if you need to balance between speed and quality.
- Make sure your RunPod instance has enough GPU memory to load the model and generate images at the specified resolution.

## Troubleshooting

- If you encounter CUDA out of memory errors, try reducing the image dimensions or using a GPU with more VRAM.
- Ensure all required packages are correctly installed in your Docker image.
- Check RunPod logs for any specific error messages if the serverless function fails to start or respond.
- If you're having issues with the conda environment or poetry dependencies, try removing and recreating the environment, or updating the dependencies in the `pyproject.toml` file.

## Other Examples:
- [Runpod's Tutorials](https://docs.runpod.io/tutorials/serverless/gpu/run-your-first)
- [Runpod's SD Tutorial](https://github.com/runpod/docs/blob/main/docs/tutorials/sdks/python/102/08-stable-diffusion-text-to-image.md)
- [Runpod's `worker-a1111`](https://github.com/runpod-workers/worker-a1111/blob/main/src/rp_handler.py)


```

curl -X POST "https://api.runpod.ai/v2/9wrmjc21x0pdlx/runsync" \
     -H "accept: application/json" \
     -H "content-type: application/json" \
     -H "authorization: ICW869VO55L7XBTG85NITQS7SG2FGFJRWYE1OVR5" \
     -d '{
        "input": {
            "prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "model_name": "black-forest-labs/FLUX.1-dev",
            "use_lora": False,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 3.5
            "seed": null
        }
     }' \
     --output "generated_image.png"

```