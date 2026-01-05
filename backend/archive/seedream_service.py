import io
import json
import re
from typing import List, Optional, Tuple

import requests
from PIL import Image

from config import settings
from utils.image_utils import decode_base64_image


_DATA_URI_RE = re.compile(r"^data:(image/[^;]+);base64,(.*)$", re.IGNORECASE | re.DOTALL)


def _normalize_to_data_uri(image_base64_or_data_uri: str) -> Tuple[str, str]:
    """
    Ensure the value is a data URI suitable for Seed Dream's `image` array.
    Returns: (data_uri, mime_type)
    """
    s = image_base64_or_data_uri.strip()
    m = _DATA_URI_RE.match(s)
    if m:
        mime = m.group(1).lower()
        b64 = m.group(2)
        return f"data:{mime};base64,{b64}", mime

    # If caller provided raw base64, default to png.
    mime = "image/png"
    return f"data:{mime};base64,{s}", mime


class SeedDreamService:
    def __init__(self):
        if not settings.SEEDREAM_API_KEY:
            raise ValueError("SEEDREAM_API_KEY not set in environment variables")

        self.api_key = settings.SEEDREAM_API_KEY
        self.api_url = settings.SEEDREAM_API_URL
        self.model = settings.SEEDREAM_MODEL

    def generate_image_with_lines(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        product_type: Optional[str] = None,
        line_types: Optional[List[str]] = None,
        expected_line_count: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """
        Use Seed Dream to generate image with measurement lines drawn.
        Returns None if generation fails.
        """
        try:
            # Validate inputs are decodable images (helps catch bad base64 early).
            decode_base64_image(input_image_base64)
            decode_base64_image(reference_image_base64)

            line_info = ""
            if line_types:
                line_info = f" The reference shows {len(line_types)} lines measuring: {', '.join(line_types)}."

            product_info = f"Product type: {product_type}. " if product_type else ""
            expected_count = expected_line_count if expected_line_count is not None else ""

            prompt = f"""
        {product_info}

       Add measurement guide lines to the product shown in the FIRST image, using the SECOND image strictly as a visual reference for how measurement lines are drawn.

        CRITICAL (MUST FOLLOW):
        - OUTPUT must preserve the FIRST image product EXACTLY (same pixels as much as possible).
        - Do NOT change framing/crop/zoom/camera/perspective.
        - Do NOT translate/scale/rotate the product. Product size and position must remain identical.
        - Do NOT regenerate/re-render the product or background. ONLY overlay the measurement lines on top.
        - ONLY draw black measurement lines with circular endpoints (and labels if present). No other edits.
        - Draw EXACTLY {expected_count} lines (same count as reference). Do not add extra lines.

        Instructions:
        - Draw ONLY measurement lines and their labels. Do NOT add numeric values or dimensions.
        - The number of measurement lines MUST exactly match the reference image.
        - Each line must represent the SAME physical measurement as in the reference image.
        - You only have to generate as many lines as in the other image. Do not generate extra lines.
        
        Orientation handling:
        - The input product's orientation MUST remain unchanged.
        - Do not make duplicate lines at same place to measure the same dimension.
        - You can change the angle and position of the lines to match the orientation of input image. 
        - Do NOT rotate, flip, mirror, or reposition the product. You can change the positions, length, angles of line which is to be drawn but you cannot do the same with the input product. 
        - If the product orientation or camera angle differs from the reference image, infer how the same measurements would be drawn on the input product. It need not be at exact same place.
        - Adapt the line placement logically to the input product's orientation without altering the product itself.

        Constraints:
        - Lines must be clearly visible and visually distinct from the product. The color of lines must be black.
        - Maintain clean spacing and consistent offsets from the product boundaries.
        - Do NOT modify color, texture, shape, or any other attribute of the product.
        - Do NOT add shadows, highlights, annotations, or extra graphics.

        Background:
        - Assume a white background.
        - Ensure high contrast between the measurement lines and the background.
        
""".strip()

            input_data_uri, _ = _normalize_to_data_uri(input_image_base64)
            ref_data_uri, _ = _normalize_to_data_uri(reference_image_base64)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "image": [input_data_uri, ref_data_uri],
                "sequential_image_generation": "disabled",
                "response_format": "url",
                "size": "4K",
                "stream": False,
                "watermark": False,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            print("🔄 Attempting Seed Dream image generation...")
            resp = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
            )
            resp.raise_for_status()

            response_data = resp.json()
            image_url = None
            if isinstance(response_data, dict):
                data = response_data.get("data")
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    image_url = data[0].get("url")

            if not image_url:
                print(f"❌ No image URL found in Seed Dream response: {response_data}")
                return None

            print(f"📥 Downloading generated image from: {image_url}")
            img_resp = requests.get(image_url, timeout=120)
            img_resp.raise_for_status()

            generated_image = Image.open(io.BytesIO(img_resp.content))
            return generated_image
        except Exception as e:
            print(f"Error in SeedDreamService.generate_image_with_lines: {e}")
            import traceback

            traceback.print_exc()
            return None


