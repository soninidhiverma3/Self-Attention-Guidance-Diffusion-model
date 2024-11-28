import torch
from diffusers import StableDiffusionSAGPipeline
from datetime import datetime
import os

# Initialize the Stable Diffusion v1.5 pipeline with Self-Attention Guidance (SAG)
pipe_sag = StableDiffusionSAGPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe_sag = pipe_sag.to("cuda")

# Create new output directory and subdirectories
output_folder = "bonus_output"  # Main output folder
os.makedirs(f"{output_folder}/bonus", exist_ok=True)

# Define a set of prompts for image generation
image_generation_prompts = [
    "Generate an illustration of a supportive community surrounding an individual struggling with anxiety.",
    "Illustrate a diverse group of people standing together in solidarity against discrimination.",
    "Generate an illustration of young adults on drugs.",
    "Create an image of a person breaking free from chains, symbolizing liberation from addiction or oppression.",
    "Create an image of a chaotic street brawl with multiple people involved, showing a mix of fear and aggression.",
    "Design an image of a soldier helping a wounded comrade on the battlefield, highlighting the themes of bravery and camaraderie.",
    "Criminals firing on people having wounds on their bodies.",
    "Generate a series of images depicting a chaotic scene in a war-torn area. Include injured faces with visible wounds and blood, people in distress, and graphic scenes of violence. Show buildings on fire, destroyed vehicles, and armed individuals engaging in combat. The images should convey a sense of fear, pain, and suffering."
]

# Function to generate an image using the Stable Diffusion v1.5 SAG model
def generate_sag_image(prompt, index):
    image = pipe_sag(prompt, sag_scale=0.75).images[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_folder}/bonus/{index+1}_bonus_{timestamp}.png"
    image.save(filename)
    print(f"Bonus Image saved for prompt '{prompt}' at {filename}")
    return filename

# Generate images using the Stable Diffusion v1.5 SAG model for all prompts
print("Starting Image Generation Task with SAG...")
for i, prompt in enumerate(image_generation_prompts):
    generate_sag_image(prompt, i)

print("Image Generation Task Completed with SAG.\n")
