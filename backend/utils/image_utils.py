import base64
import io
from PIL import Image
import numpy as np

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[-1]
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def encode_image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """Encode PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def resize_image(image: Image.Image, max_size: int = 2048) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def resize_to_square(image: Image.Image, size: int = 1024) -> Image.Image:
    """
    Resize image to square (size x size) maintaining aspect ratio with white padding
    Returns 1024x1024 image with original image centered
    """
    width, height = image.size
    
    # If already square and correct size, return as is
    if width == size and height == size:
        return image
    
    # Calculate scaling to fit within square while maintaining aspect ratio
    scale = min(size / width, size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new square image with white background
    square_image = Image.new('RGB', (size, size), (255, 255, 255))
    
    # Calculate position to center the resized image
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    
    # Paste resized image onto white square
    square_image.paste(resized, (x_offset, y_offset))
    
    return square_image

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array"""
    return np.array(image.convert('RGB'))

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image"""
    return Image.fromarray(arr.astype('uint8'), 'RGB')

def expand_canvas_to_aspect_ratio(image: Image.Image, target_aspect_ratio: float) -> Image.Image:
    """
    Expand image canvas with white padding to match target aspect ratio.
    Only expansion is allowed (no cropping). The original image is centered on the new canvas.
    
    Args:
        image: PIL Image to expand
        target_aspect_ratio: Target aspect ratio (width / height) to match
        
    Returns:
        Expanded PIL Image with white background, matching target aspect ratio
    """
    img_width, img_height = image.size
    img_aspect = img_width / img_height
    
    # Calculate minimum canvas dimensions that:
    # 1. Match target_aspect_ratio
    # 2. Are >= image dimensions in both width and height
    
    if img_aspect > target_aspect_ratio:
        # Generated image is wider relative to target
        # Try fixing width first
        canvas_width = img_width
        canvas_height = int(img_width / target_aspect_ratio)
        
        # If calculated height is less than image height, fix height instead
        if canvas_height < img_height:
            canvas_height = img_height
            canvas_width = int(img_height * target_aspect_ratio)
    else:
        # Generated image is taller relative to target
        # Try fixing height first
        canvas_height = img_height
        canvas_width = int(img_height * target_aspect_ratio)
        
        # If calculated width is less than image width, fix width instead
        if canvas_width < img_width:
            canvas_width = img_width
            canvas_height = int(img_width / target_aspect_ratio)
    
    # Create white canvas
    expanded_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    # Calculate position to center the original image
    x_offset = (canvas_width - img_width) // 2
    y_offset = (canvas_height - img_height) // 2
    
    # Paste original image onto white canvas
    expanded_image.paste(image, (x_offset, y_offset))
    
    return expanded_image

