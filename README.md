@author: Chibundum Adebayo
@creation-date: 06.01.2025

# Visually-Grounded Question Answering with Foundation Models

## Introduction

## Project Structure

```
grounded-vqa-fm/
├── data/
│   ├── vqa-v2/
│   │   ├── train2014/
│   │   ├── val2014/
│   │   ├── questions/
│   │   ├── annotations/
│   ├── vcr1annots/
│   ├── vcr1images/
├── zero_shot_clip/
│   ├──clip_no_answer.py
│   ├──clip_answer.py
├── linear_clip/
│   ├── linear_clip.py
├── cross_modal_clip/
│   ├── model.py
│   ├── cross_modal_fusion.py
├── zero_shot.py
├── linear_train.py
├── cma_train.py
├── results/
```

## Project Setup

### 1. Create a virtual environment using either `venv` or `conda` and install the required packages.

```python
# With Virtual Environment
python3 -m venv vqa_env
source vqa_env/bin/activate
pip install -r requirements.txt

# With Conda Environment
conda create --name vqa_env --file requirements.txt
conda activate vqa_env
```

### 2. Clone the repository and navigate to the project directory.

```bash
git clone https://github.com/ipinmi/grounded-vqa-fm.git
```

### 3. Download the required datasets and pre-trained models.

a. VCR dataset

```bash
# Download the VCR dataset and annotations from the project's Kaggle dataset
# Add your Kaggle API key as an environment variable

# https://visualcommonsense.com/download/

pip install kaggle --quiet
kaggle datasets download ipinmi/visual-question-and-answering
unzip -q visual-question-and-answering.zip -d data
```

b. VQA v2 dataset

```bash
# Download the VQA v2 dataset
mkdir -p data/vqa_v2
curl -OL http://images.cocodataset.org/zips/train2014.zip OR wget http://images.cocodataset.org/zips/train2014.zip
curl -OL http://images.cocodataset.org/zips/val2014.zip OR wget http://images.cocodataset.org/zips/val2014.zip

# the VQA v2 annotations and questions are included in the created kaggle dataset
unzip train2014.zip -d data/vqa_v2
unzip val2014.zip -d data/vqa_v2
```

c. Install the CLIP model (copied from the [official repository](https://github.com/openai/CLIP))

```bash
# First, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.
```

d. Install the libraries for the Transformer-MM-Explainability repository

```bash

source: https://github.com/hila-chefer/Transformer-MM-Explainability

# cd Transformer-MM-Explainability

pip install einops
pip install captum
pip install opencv-python
```
