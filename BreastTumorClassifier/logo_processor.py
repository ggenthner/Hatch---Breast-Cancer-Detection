from PIL import Image
import os

def process_logo():
    # Open the original logo
    original_logo = Image.open("generated-icon.png").convert('RGBA')

    # Create a white background image
    white_bg = Image.new('RGBA', original_logo.size, (255, 255, 255, 255))

    # Composite the images together
    output_image = Image.alpha_composite(white_bg, original_logo)

    # Convert to RGB before saving to ensure no transparency
    output_image = output_image.convert('RGB')

    # Save with high quality
    output_image.save("generated-icon.png", "PNG", quality=95)

if __name__ == "__main__":
    process_logo()