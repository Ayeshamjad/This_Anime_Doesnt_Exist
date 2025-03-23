import torch
import torchvision.utils as vutils
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from loadmodel import load_generator  # Import the preloaded model

# Load model once
netG, device = load_generator(use_cuda=False)  # Set to True for CUDA

# Streamlit UI
st.title("🎨 This Anime Doesnt Exists")
st.write("Click the button below to generate a new image.")

# Function to generate an image
def generate_image():
    noise = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_image = netG(noise).detach().cpu()

    # Convert tensor to PIL image with correct normalization
    img_grid = vutils.make_grid(fake_image, padding=2, normalize=True)
    img_pil = Image.fromarray((img_grid.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    # Save to buffer (keeping natural size)
    img_io = BytesIO()
    img_pil.save(img_io, format="PNG")
    img_io.seek(0)
    
    return img_io

# Button to generate an image
if st.button("✨ Generate Image"):
    with st.spinner("Generating..."):
        img = generate_image()
        st.image(img, caption="AI-Generated Image", use_container_width=False)  # Natural size
