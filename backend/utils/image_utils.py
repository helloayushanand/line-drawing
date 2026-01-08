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

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array"""
    return np.array(image.convert('RGB'))

def expand_canvas_to_aspect_ratio(image: Image.Image, target_aspect_ratio: float) -> Image.Image:
    """Expand image canvas with white padding to match target aspect ratio"""
    img_width, img_height = image.size
    img_aspect = img_width / img_height
    
    if img_aspect > target_aspect_ratio:
        canvas_width = img_width
        canvas_height = int(img_width / target_aspect_ratio)
        
        if canvas_height < img_height:
            canvas_height = img_height
            canvas_width = int(img_height * target_aspect_ratio)
    else:
        canvas_height = img_height
        canvas_width = int(img_height * target_aspect_ratio)
        
        if canvas_width < img_width:
            canvas_width = img_width
            canvas_height = int(img_width / target_aspect_ratio)
    
    expanded_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    x_offset = (canvas_width - img_width) // 2
    y_offset = (canvas_height - img_height) // 2
    expanded_image.paste(image, (x_offset, y_offset))
    
    return expanded_image

