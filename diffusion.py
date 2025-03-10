# Install necessary libraries if not already installed
# pip install torch torchvision torchaudio diffusers transformers Pillow

# Import required libraries
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os


def load_model():
    """Load the Stable Diffusion model."""
    print("Loading Stable Diffusion model...")
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded successfully.")
    return model


def generate_image(model, description: str, file_location: str):
    """Generate an image based on the text description and save it."""
    try:
        # Ensure the directory exists
        os.makedirs(file_location, exist_ok=True)

        # Generate the image
        print(f"Generating image for prompt: {description}")
        image = model(description).images[0]

        # Define file path
        image_path = os.path.join(file_location, "generated_image.png")

        # Save the generated image
        image.save(image_path)
        print(f"Image saved at: {image_path}")

        # Show the generated image
        image.show()

        return image_path

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None


if __name__ == "__main__":
    # Load the model once
    model = load_model()

    # Get user input
    description = input("Enter description to generate an image: ")

    # Set the file location (local directory)
    file_location = os.path.join(os.getcwd(), "GeneratedImages")  # Saves in a folder within the current directory

    # Generate and save the image
    image_path = generate_image(model, description, file_location)

    # Output the image path
    if image_path:
        print(f"Generated image saved at: {image_path}")
