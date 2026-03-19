from PIL import Image

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