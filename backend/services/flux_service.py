import io
import json
import time
from typing import List, Optional

import requests
from PIL import Image

from config import settings


def _strip_data_uri_prefix(image_base64_or_data_uri: str) -> str:
    s = image_base64_or_data_uri.strip()
    if "," in s and s.lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


class FluxService:
    def __init__(self):
        if not settings.BFL_API_KEY:
            raise ValueError("BFL_API_KEY not set in environment variables")

        self.api_key = settings.BFL_API_KEY
        self.api_url = settings.FLUX_API_URL

    def generate_image_with_lines(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        prompt: str,
        guidance : int = 10,
        timeout_seconds: int = 180,
        poll_interval_seconds: float = 1.0,
    ) -> Optional[Image.Image]:
        """
        Call Flux 2 Pro with two input images (raw base64, no data URI prefix).
        Uses polling_url until Ready, then downloads result.sample.
        """
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
            print(f"Error in FluxService.generate_image_with_lines: {e}")
            import traceback

            traceback.print_exc()
            return None


