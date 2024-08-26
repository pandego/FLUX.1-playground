# FLUX.1-playground
A playground for the new FLUX.1 models, for both inference and training LoRas.


## 🚀 Quick Start

Follow these steps to clone the repository, set up the environment, and start using FLUX.1 models:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/pandego/FLUX.1-playground.git
   cd FLUX.1-playground
   ```

2. **Set Up the Environment:**

   Create the conda environment with all necessary dependencies:

   ```bash
   conda env create -f environment.yml
   conda activate ai-toolkit
   ```

3. **Install Dependencies:**

   Use Poetry to install additional dependencies:

   ```bash
   poetry install --no-root
   ```

4. **Clone [ai-toolkit](https://github.com/ostris/ai-toolkit):**

   Clone the ai-toolkit repository directly into your playground:

   ```bash
   git clone https://github.com/ostris/ai-toolkit.git
   cd ai-toolkit
   git submodule update --init --recursive
   cd ..
   ```

## Inference using FLUX.1 models

   ```bash
   cd ..
   python run_flux_inference.py
   ```

---
## Training using [ai-toolkit](https://github.com/ostris/ai-toolkit)
# WIP
---

## 🤝 Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

## 📄 Acknowledgments

Special thanks to the developers of [ai-toolkit](https://github.com/ostris/ai-toolkit) and other contributors for providing the tools and inspiration to build this playground.

---

_Et Voilà !_ 🎈