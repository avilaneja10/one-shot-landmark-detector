from PIL import Image
from huggingface_hub import login
import os
import logging
import math

logger = logging.getLogger(__name__)

def load_image(path: str) -> Image.Image:
    """Load an image as RGB PIL Image."""
    return Image.open(path).convert("RGB")

def pad_to_multiple(image: Image.Image, multiple: int) -> Image.Image:
    """
    Pad image with black pixels on the right and bottom so that both
    dimensions are exact multiples of `multiple`.
 
    If already aligned, returns the original image unchanged.
    """
    w, h = image.size
    pad_right = (multiple - w % multiple) % multiple
    pad_bottom = (multiple - h % multiple) % multiple
 
    if pad_right == 0 and pad_bottom == 0:
        return image
 
    padded = Image.new("RGB", (w + pad_right, h + pad_bottom), (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded

def login_huggingface():
    """
    Authenticate with Hugging Face Hub.
 
    Checks for a token in the HF_TOKEN environment variable first.
    If not found, prompts the user interactively.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face using HF_TOKEN environment variable")
    else:
        logger.info("HF_TOKEN not found in environment, prompting for login")
        login()

def get_image_dims(img_path: str) -> tuple[int, int]:
    """Read image dimensions from file header without loading pixels."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)

def xy_to_index(x, y, orig_w, orig_h, level, patch_size=16):
    """
    Convert pixel coordinate (x, y) to flat index in the embedding tensor.
    This is required to compare the final results and get the embeddings corresponding to
    a coordinate.
    
    Args:
        x, y: Pixel coordinates in the original image.
        orig_w, orig_h: Original image dimensions.
        level: "patch" or "pixel".
        patch_size: ViT patch size (only used for patch level).

    Returns:
        Flat index into the (K, D) embedding tensor.
    """
    if level == "pixel":
        return x * orig_h + y
    else:
        grid_cols = math.ceil(orig_w / patch_size)
        return (y // patch_size) * grid_cols + (x // patch_size)