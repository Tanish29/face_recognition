# Face Recognition in Python (Ongoing)
This repository contains code that loads in the `CELEBA` dataset and trains a deep learning model to identiy the trained faces to a high accuracy. Furthermore after training the best model is validated. 

There is a main file that performs the key steps in a deep learning project. To run that file create a local clone of the repository and follow the instructions below.

## Installation
Prior to running the main script you will need to install the dependencies. The preferred python version is 3.12.

This project supports [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager and is recommeneded due to its fast package installation. However instructions for `pip` are provided.

### UV
1. Ensure `uv` is installed via the provided link
   e.g., to install on linux/macos run:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. To install deps run:
   ```
   uv sync --locked
   ```

### PIP
1. Create a virtual env. 
   e.g., using [conda](https://docs.conda.io/en/latest/) run:
   ```
   conda create -n env_name python=3.12
   ```
2. Activate environment and install deps:
   ```
   pip install .
   ```

## Dataset
Next to download and process the CELEBA dataset simply run the build script via this command:
```
python -m tools.build_celeba_dataset --out-dir dataset/celeba
```
**NB: This may take a couple of minutes.**

