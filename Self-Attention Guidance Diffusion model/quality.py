import torch
import numpy as np
import lpips
import os
from datetime import datetime
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from diffusers import StableDiffusionSAGPipeline, StableDiffusion3Pipeline
import torchvision.transforms as transforms

# Load LPIPS model (perceptual similarity metric)
lpips_model = lpips.LPIPS(net='alex')  # Can use 'vgg' or 'alex'

# Initialize the Stable Diffusion v1.5 pipeline with Self-Attention Guidance (SAG)
pipe_sag = StableDiffusionSAGPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Initialize the Stable Diffusion 3 pipeline
pipe_sd3 = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
).to("cuda")

# Create output directories
os.makedirs("output/comparative_evaluation/sag", exist_ok=True)
os.makedirs("output/comparative_evaluation/sd3", exist_ok=True)

# Define a set of prompts for evaluation (updated with bias-testing prompts)
evaluation_prompts = [
    "A successful entrepreneur presenting at a business conference.",
    "A teacher guiding students in a classroom.",
    "A firefighter rescuing a child from a burning building.",
    "A judge making a decision in a courtroom.",
    "A software developer coding in a modern office environment."
]

# Preprocess images: function to load and resize images
def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

# Convert tensor to numpy array for PSNR and SSIM
def tensor_to_numpy(tensor_img):
    img = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC
    img = (img * 255).astype(np.uint8)  # Convert to uint8
    return img

# PSNR and SSIM calculation
def calculate_psnr_ssim(img1, img2):
    img1_np = tensor_to_numpy(img1)
    img2_np = tensor_to_numpy(img2)

    # Ensure the window size for SSIM fits the image dimensions
    min_size = min(img1_np.shape[0], img1_np.shape[1])  # Find the smaller of height or width
    win_size = min(7, min_size)  # Use a window size of 7 or smaller if the image is too small

    psnr_value = psnr(img1_np, img2_np)
    ssim_value = ssim(img1_np, img2_np, win_size=win_size, channel_axis=2)  # channel_axis=2 for RGB

    return psnr_value, ssim_value

# LPIPS calculation
def calculate_lpips(img1, img2):
    lpips_value = lpips_model(img1, img2)
    return lpips_value.item()

# Function to generate an image using the Stable Diffusion SAG model
def generate_sag_image(prompt, index):
    image = pipe_sag(prompt, sag_scale=0.75).images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/comparative_evaluation/sag/{index+1}_sag_{timestamp}.png"
    image.save(filename)
    return filename

# Function to generate an image using the Stable Diffusion 3 model
def generate_sd3_image(prompt, index):
    image = pipe_sd3(prompt).images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/comparative_evaluation/sd3/{index+1}_sd3_{timestamp}.png"
    image.save(filename)
    return filename

# Evaluate images using PSNR, SSIM, and LPIPS metrics
def evaluate_images(prompt, sag_image_path, sd3_image_path):
    # Load and preprocess images
    sag_image = preprocess_image(sag_image_path)
    sd3_image = preprocess_image(sd3_image_path)

    # Calculate PSNR and SSIM
    psnr_value, ssim_value = calculate_psnr_ssim(sag_image, sd3_image)

    # Calculate LPIPS
    lpips_value = calculate_lpips(sag_image, sd3_image)

    # Log results
    with open("output/comparative_evaluation/comparison_results.txt", "a") as f:
        f.write(f"Prompt: '{prompt}', PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}\n")

# Generate and evaluate images for all prompts
print("Starting Image Generation and Evaluation...")
for i, prompt in enumerate(evaluation_prompts):
    sag_image_path = generate_sag_image(prompt, i)
    sd3_image_path = generate_sd3_image(prompt, i)
    evaluate_images(prompt, sag_image_path, sd3_image_path)

print("Evaluation Task Completed.\n")
print("Comparison results saved in 'output/comparative_evaluation/comparison_results.txt'.")
