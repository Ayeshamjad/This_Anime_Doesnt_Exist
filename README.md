# This Anime Doesn't Exist - GAN-based Anime Character Generation

## Overview
This project implements a **Generative Adversarial Network (GAN)** trained on an anime dataset to generate unique anime-style characters. The model is built using **PyTorch** and is deployed as a web application using **Streamlit**, hosted on Hugging Face Spaces.

## Features
- **Deep Learning-based Image Generation**: Uses a GAN architecture to generate high-quality anime faces.
- **Streamlit UI**: A simple and interactive interface for generating and downloading AI-generated anime images.
- **Efficient Deployment**: Hosted on Hugging Face Spaces for accessibility.

## Installation & Running Locally
### Prerequisites
Ensure you have Python 3.10+ installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ayeshamjad/This_Anime_Doesnt_Exist
cd this_anime_doesnt_exist
```

### Step 2: Install Dependencies
```bash
pip install -r requirement.txt
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

## Training the Model
The model was trained on an anime face dataset using a **DCGAN architecture**. The training script includes:
- Data preprocessing (image resizing & normalization)
- Model architecture definition
- Training loop with loss visualization
- Saving the trained generator

## Generating Anime Faces
1. Visit the **[Live App](https://huggingface.co/spaces/ayesha016/This_Anime_Doesnt_Exist**.
2. Click the "Generate Image" button.


