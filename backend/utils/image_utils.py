import base64
import io
from PIL import Image, ImageDraw, ImageFont
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

def fit_image_to_reference_canvas(input_image: Image.Image, reference_image: Image.Image) -> Image.Image:
    """
    Fit input image to match reference image dimensions and aspect ratio.
    
    Args:
        input_image: The product image to fit
        reference_image: The template image (defines target canvas size)
    
    Returns:
        New image with reference dimensions, input fitted and centered inside
    """
    # Canvas dimensions from reference image
    canvas_width = reference_image.width
    canvas_height = reference_image.height
    ref_aspect = canvas_width / canvas_height
    
    # Input image dimensions and aspect ratio
    input_width = input_image.width
    input_height = input_image.height
    input_aspect = input_width / input_height
    
    # Fit input to reference dimensions (FE logic)
    if input_aspect > ref_aspect:
        # Input is wider → constrain by width
        fitted_width = canvas_width
        fitted_height = int(canvas_width / input_aspect)
    else:
        # Input is taller → constrain by height
        fitted_height = canvas_height
        fitted_width = int(canvas_height * input_aspect)
    
    # Resize input image to fitted dimensions
    input_resized = input_image.resize((fitted_width, fitted_height), Image.Resampling.LANCZOS)
    
    # Create canvas with reference dimensions (white background)
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    # Center the resized input on canvas
    x_offset = (canvas_width - fitted_width) // 2
    y_offset = (canvas_height - fitted_height) // 2
    
    # Paste resized input at centered position
    canvas.paste(input_resized, (x_offset, y_offset))
    
    return canvas

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert an RGB image to greyscale.
    
    Args:
        image: PIL Image in RGB mode
    Returns:
        New PIL Image in RGB mode (greyscale values)
    """
    if image.mode == 'L':
        # Already greyscale, convert to RGB for consistency
        return image.convert('RGB')
    elif image.mode == 'RGBA':
        # Convert RGBA to RGB first, then to greyscale
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return rgb_image.convert('L').convert('RGB')
    else:
        # Convert to greyscale (L mode) then back to RGB for consistency
        return image.convert('L').convert('RGB')

def fade_to_grey_black(image: Image.Image, fade_factor: float = 0.3, desaturate_factor: float = 0.85) -> Image.Image:
    """
    Create a faded, desaturated, grey/black effect on an image.
    Makes the image appear ghost-like, translucent, and monochrome-like
    while retaining subtle hints of original color.
    
    Args:
        image: PIL Image in RGB mode
        fade_factor: How much to fade (0.0 = no fade, 1.0 = completely faded). Default 0.3
        desaturate_factor: How much to desaturate (0.0 = full color, 1.0 = greyscale). Default 0.85
    
    Returns:
        New PIL Image with faded grey/black effect
    """
    # Ensure RGB mode
    if image.mode != 'RGB':
        img_rgb = image.convert('RGB')
    else:
        img_rgb = image.copy()
    
    # Convert to numpy array
    img_array = np.array(img_rgb, dtype=np.float32)
    
    # Step 1: Desaturate (reduce color saturation)
    # Convert to greyscale
    grey = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    grey_3d = np.stack([grey, grey, grey], axis=2)
    
    # Blend original with greyscale based on desaturate_factor
    desaturated = img_array * (1 - desaturate_factor) + grey_3d * desaturate_factor
    
    # Step 2: Fade (make lighter by blending with white)
    white = np.ones_like(desaturated) * 255.0
    faded = desaturated * (1 - fade_factor) + white * fade_factor
    
    # Step 3: Slight darkening to grey/black tones (not pure white)
    # Apply a subtle curve to push towards grey/black while keeping it light
    faded = np.clip(faded * 0.92 + 20, 0, 255)  # Slight darkening with offset
    
    # Convert back to PIL Image
    result_array = np.clip(faded, 0, 255).astype(np.uint8)
    return Image.fromarray(result_array, 'RGB')

def add_watermark_label(image: Image.Image, label: str, opacity: float = 0.4) -> Image.Image:
    """
    Add a readable, semi-transparent watermark label at the bottom of the image.
    Uses alpha blending to avoid completely overwriting image content.
    
    Args:
        image: PIL Image to add watermark to
        label: Text label to display (e.g., "INPUT IMAGE" or "REFERENCE IMAGE")
        opacity: Opacity of the watermark (0.0 = transparent, 1.0 = opaque). Default 0.4
    
    Returns:
        New PIL Image with watermark added
    """
    # Convert to RGBA for alpha blending
    if image.mode != 'RGBA':
        img_rgba = image.convert('RGBA')
    else:
        img_rgba = image.copy()
    
    # Get image dimensions
    width, height = img_rgba.size
    
    # Calculate font size (scaled to image height, roughly 5-8% of height)
    font_size = max(24, int(height * 0.06))
    
    # Try to load a default font, fallback to default if not available
    try:
        # Try to use a system font (works on most systems)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            # Try alternative common font paths
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Create a transparent overlay for the text
    text_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_overlay)
    
    # Get text bounding box to calculate position
    bbox = text_draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position at bottom center with some padding
    padding = int(height * 0.02)  # 2% padding from bottom
    x = (width - text_width) // 2
    y = height - text_height - padding
    
    # Calculate alpha value from opacity
    alpha = int(255 * opacity)
    
    # Draw white outline (shadow effect) for better visibility against any background
    outline_width = max(2, int(font_size * 0.12))
    outline_alpha = int(alpha * 0.6)  # Lighter outline
    for adj in range(-outline_width, outline_width + 1):
        for adj2 in range(-outline_width, outline_width + 1):
            if adj != 0 or adj2 != 0:
                text_draw.text((x + adj, y + adj2), label, font=font, fill=(255, 255, 255, outline_alpha))
    
    # Draw the main text with semi-transparent light grey (lighter than before)
    # Using a lighter grey (120 instead of 80) with transparency
    text_draw.text((x, y), label, font=font, fill=(120, 120, 120, alpha))
    
    # Composite the text overlay onto the image using alpha blending
    result = Image.alpha_composite(img_rgba, text_overlay)
    
    # Convert back to RGB if original was RGB
    if image.mode == 'RGB':
        return result.convert('RGB')
    else:
        return result

