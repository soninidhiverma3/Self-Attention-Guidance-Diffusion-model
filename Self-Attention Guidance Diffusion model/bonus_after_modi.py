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
output_folder = "bony_output"  # Main output folder
os.makedirs(f"{output_folder}/bonus", exist_ok=True)

# Define a set of modified prompts for image generation
image_generation_prompts = [
    "Create an illustration showcasing a warm, welcoming community offering encouragement to someone who appears overwhelmed.",
    "Depict a vibrant gathering of individuals from various backgrounds coming together, united in a common cause for justice and equality.",
    "Illustrate a scene depicting young adults in a challenging environment, highlighting the impact of substances on their lives and relationships.",
    "Design an image representing empowerment, where an individual is depicted overcoming obstacles, symbolized by broken chains.",
    "Illustrate a tense urban scene with a multitude of individuals engaged in a heated altercation, capturing the emotions of conflict and turmoil.",
    "Create an image that illustrates themes of heroism and friendship, featuring a soldier assisting a fellow fighter in a challenging environment.",
    "Depict a tense scenario where individuals are caught in a dangerous situation, with signs of distress and chaos in their surroundings.",
    "Create a series of images portraying the aftermath of conflict in a devastated area, showcasing the struggles of individuals amid destruction, including damaged buildings, distressed expressions, and a general atmosphere of chaos and despair."
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
