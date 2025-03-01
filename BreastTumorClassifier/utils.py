from PIL import Image
import io

def is_valid_image(file_upload):
    """
    Validate if the uploaded file is a valid image
    """
    try:
        image = Image.open(file_upload)
        image.verify()
        return True
    except Exception:
        return False

def preprocess_image(image):
    """
    Preprocess the image for the model
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if needed (adjust size based on model requirements)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        image = image.resize(new_size, Image.LANCZOS)
    
    return image
