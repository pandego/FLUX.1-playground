FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04


# ----------------------------------- #
# --- Set up the base environment --- #
# ----------------------------------- #

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1


# ---------------------------------- #
# --- Install necessary packages --- #
# ---------------------------------- #

RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    && apt clean && rm -rf /var/lib/apt/lists/*


# ------------------------------------ #
# --- Setup Python Env with Poetry --- #
# ------------------------------------ #

RUN pip install --no-cache-dir poetry --verbose

WORKDIR /src

COPY pyproject.toml .

RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction --no-ansi --only main -vvv

RUN pip install --no-deps --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl'


# ---------------------------------- #
# --- Preload HuggingFace models --- #
# ---------------------------------- #

# COPY ./huggingface /root/.cache/huggingface

# Alternatively, you can preload models from HuggingFace by downloading them to cache
# RUN --mount=type=secret,id=env \
#     export HF_TOKEN=$(cat /run/secrets/env) && \
#     python -c "from diffusers import FluxPipeline; \
#     FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev')"
#     python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('black-forest-labs/FLUX.1-dev')"
# This will only work if you use '--secret' to build the image
# docker build --secret id=env,env=HF_TOKEN -t flux_image_generator .


# ---------------------------- #
# --- Start the Serverless --- #
# ---------------------------- #

COPY handler.py handler.py

CMD ["python", "-u", "handler.py"]
