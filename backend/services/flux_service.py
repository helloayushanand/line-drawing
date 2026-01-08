import io
import json
import time
import base64
from typing import List, Optional

import requests
from PIL import Image

from config import settings


def _strip_data_uri_prefix(image_base64_or_data_uri: str) -> str:
    s = image_base64_or_data_uri.strip()
    if "," in s and s.lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _jpeg_data_uri_under_1mb(img: Image.Image, max_bytes: int = 1_000_000) -> str:
    """Encode an image as JPEG data URI <= max_bytes by reducing quality"""
    for q in (90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
    raise RuntimeError("Could not compress image to <= 1MB at 1024x1024")


class FluxService:
    def __init__(self):
        if not settings.BFL_API_KEY:
            raise ValueError("BFL_API_KEY not set in environment variables")

        self.api_key = settings.BFL_API_KEY
        self.api_url = settings.FLUX_API_URL

    def _generate_image_with_prompt(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        prompt: str,
        guidance : int = 10,
        timeout_seconds: int = 180,
        poll_interval_seconds: float = 1.0,
    ) -> Optional[Image.Image]:
        """Call Flux 2 Pro with two input images and a prompt"""
        try:
            input_b64 = _strip_data_uri_prefix(input_image_base64)
            ref_b64 = _strip_data_uri_prefix(reference_image_base64)

            resp = requests.post(
                self.api_url,
                headers={"x-key": self.api_key},
                json={
                    "prompt": prompt,
                    "input_image": input_b64,
                    "input_image_2": ref_b64,
                    "guidance": guidance,
                },
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()

            polling_url = payload.get("polling_url")
            if not polling_url:
                print(f"❌ Flux response missing polling_url: {payload}")
                return None

            start = time.time()
            while True:
                if time.time() - start > timeout_seconds:
                    print("❌ Flux polling timed out")
                    return None

                result = requests.get(
                    polling_url,
                    headers={"accept": "application/json", "x-key": self.api_key},
                    timeout=60,
                )
                result.raise_for_status()
                result_json = result.json()

                status = result_json.get("status")
                if status == "Ready":
                    sample_url = (
                        (result_json.get("result") or {}).get("sample")
                        if isinstance(result_json.get("result"), dict)
                        else None
                    )
                    if not sample_url:
                        print(f"❌ Flux Ready response missing result.sample: {result_json}")
                        return None
                    img_resp = requests.get(sample_url, timeout=120)
                    img_resp.raise_for_status()
                    return Image.open(io.BytesIO(img_resp.content))

                if status == "Error":
                    print(f"❌ Flux returned Error: {result_json}")
                    return None

                time.sleep(poll_interval_seconds)

        except Exception as e:
            print(f"Error in FluxService._generate_image_with_prompt: {e}")
            import traceback

            traceback.print_exc()
            return None

    def generate_image_with_lines(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        product_type: Optional[str] = None,
        line_types: Optional[List[str]] = None,
        expected_line_count: Optional[int] = None,
        reference_lines: Optional[List[dict]] = None
    ) -> Optional[Image.Image]:
        """Generate image with measurement lines using Flux 2 Pro"""
        from utils.image_utils import decode_base64_image
        
        input_img = decode_base64_image(input_image_base64)
        ref_img = decode_base64_image(reference_image_base64)
        
        input_image_base64_compressed = _jpeg_data_uri_under_1mb(input_img, max_bytes=1_000_000)
        reference_image_base64_compressed = _jpeg_data_uri_under_1mb(ref_img, max_bytes=1_000_000)
        
        prompt = (
            f"{product_type or ''}\n"
            f"{'' if not line_types else 'Line types: ' + ', '.join(line_types)}\n"
            "Add measurement guide lines to the product in the FIRST image using the SECOND image as reference. "
            f"Draw EXACTLY {expected_line_count if expected_line_count is not None else 'the same number of'} measurement lines as in the reference image. "
            "Do NOT draw extra lines.\n"
            "Only draw measurement lines (black) with circular endpoints. Do not change the product."
            "You are not allowed to draw more number of lines than the reference image. If the reference image has 2 measurement lines you will draw only two, if it has 3 or 4 or 5 you will draw similar number of measurement lines."
            """

                    ====================================================================
                    VISUAL OUTPUT REQUIREMENTS (HARD CONSTRAINTS)
                    ====================================================================

                    BACKGROUND:
                    - Pure white only (RGB 255, 255, 255)

                    PRODUCT:
                    - Drawn as a VERY FAINT, ghost-like outline only
                    - Color: light gray (#E0E0E0)
                    - No visible edge lines, no fill, no shading

                    MEASUREMENT LINES:
                    - Color: Pure black (#000000)
                    - Style: Solid lines with dot endpoints
                    - Every black line MUST represent a real measurement from the reference
                    - The number of lines must be same as in the reference image.

                    IF THE ORIENTATION OF REFERENCE IMAGE AND INPUT IMAGE IS NOT THE SAME THEN :

                    - The lines in the output image should be rotated to match the orientation of the reference image. Try figuring out the correct angle and placement of the lines to be drawn based on the reference image. It should make sense.

                    ADDITIONAL INSTRUCTIONS:

                    - DO NOT draw any extra lines. The number of lines must be same and they must represnt the same measurements as in the reference image.
                    - DO not write the dimenisions. The job is to draw lines only. Not write dimensions.


                    Things to avoid:
                    - drawing unnecessary lines
                    - drawing lines that are not present in the reference image
                    - Do not draw lines on second image, your job is to draw lines on first image.
                    """
        ).strip()
        
        print("\n📸 Attempting Flux 2 Pro image generation...")
        return self._generate_image_with_prompt(
            input_image_base64_compressed,
            reference_image_base64_compressed,
            prompt
        )


