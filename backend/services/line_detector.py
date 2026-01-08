import time
import base64
import io
from typing import List, Tuple, Optional
from PIL import Image
from services.gemini_service import GeminiService
# from services.seedream_service import SeedDreamService # Archived
from services.flux_service import FluxService
from services.coordinate_extractor import CoordinateExtractor
from services.ocr_service import OCRService  # NEW
from utils.image_utils import decode_base64_image, pil_to_numpy, resize_to_square, encode_image_to_base64, resize_image, expand_canvas_to_aspect_ratio
from utils.edge_detection import detect_edges, snap_point_to_edge
from models.schemas import LineDrawing, Point
from config import settings
import numpy as np

def _jpeg_data_uri_under_1mb(img: Image.Image, max_bytes: int = 1_000_000) -> str:
    """
    Encode an image as JPEG data URI <= max_bytes by reducing quality.
    Raises if it cannot meet the limit at 1024x1024.
    """
    # Use a descending quality ladder; optimize helps reduce size.
    for q in (90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode("utf-8")
    raise RuntimeError("Could not compress image to <= 1MB at 1024x1024")

class LineDetector:
    def __init__(self):
        self.gemini = GeminiService()
        # self.seedream = SeedDreamService() # Archived
        try:
            self.flux = FluxService()
        except ValueError:
            self.flux = None
            print("⚠️  Flux service not available (BFL_API_KEY not set)")
        self.extractor = CoordinateExtractor()
        self.ocr = OCRService()  # NEW: Initialize OCR service
    
    def detect_lines(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        product_type: str = None,
        expected_line_count: int = None,
        line_types: List[str] = None,
        model: str = "gemini",
        reference_lines: List[dict] = None,  # NEW: Labeled reference lines
    ) -> Tuple[List[LineDrawing], str, float, Optional[str], Optional[str]]:
        """
        Main method: Detect lines using image generation (Gemini or Seed Dream)
        Returns: (lines, method_used, confidence_score, generated_image_base64, input_image_base64)
        """
        start_time = time.time()
        
        # Decode images
        input_img = decode_base64_image(input_image_base64)
        ref_img = decode_base64_image(reference_image_base64)
        
        # Resize images to max 1024 dimension maintaining aspect ratio
        print("🔄 Resizing images to max 1024px for consistent processing...")
        input_img_square = resize_image(input_img, 1024)
        ref_img_square = resize_image(ref_img, 1024)
        print(f"✓ Input image resized: {input_img.size} -> {input_img_square.size}")
        print(f"✓ Reference image resized: {ref_img.size} -> {ref_img_square.size}")
        
        # Convert resized images back to base64 (data URIs)
        input_base64_square = encode_image_to_base64(input_img_square)
        ref_base64_square = encode_image_to_base64(ref_img_square)
        input_image_base64_square = f"data:image/png;base64,{input_base64_square}"
        reference_image_base64_square = f"data:image/png;base64,{ref_base64_square}"
        
        model_name = (model or "gemini").lower()
        generated_image = None

        if model_name == "flux":
            if not self.flux:
                raise RuntimeError(
                    "Flux service is not available. Please set BFL_API_KEY environment variable."
                )
            # Flux input cap: ensure both images are <= 1MB each while keeping 1024x1024 pixels.
            input_image_base64_square = _jpeg_data_uri_under_1mb(input_img_square, max_bytes=1_000_000)
            reference_image_base64_square = _jpeg_data_uri_under_1mb(ref_img_square, max_bytes=1_000_000)

            print("\n📸 Attempting Flux 2 Pro image generation...")
            generated_image = self.flux.generate_image_with_lines(
                input_image_base64_square,
                reference_image_base64_square,
                prompt=(
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

                ).strip(),
            )
        # elif model_name == "seedream":
            # ... (Archived)

        else:
            print("\n📸 Attempting Gemini 3 image generation...")
            generated_image = self.gemini.generate_image_with_lines(
                input_image_base64_square,
                reference_image_base64_square,
                product_type,
                line_types,
                expected_line_count=expected_line_count,
                reference_lines=reference_lines,  # NEW: Pass labeled reference lines
            )
        
        if not generated_image:
            raise RuntimeError(
                f"{model_name} image generation failed. No image was generated. "
                "Please check your API key and that the selected model supports image generation."
            )
        
        # Ensure generated image is within max bounds (handling aspect ratio)
        if max(generated_image.size) > 1024:
            print(f"⚠️  Generated image size is {generated_image.size}, resizing to max 1024...")
            generated_image = resize_image(generated_image, 1024)
        
        # Expand generated image canvas to match reference image aspect ratio
        ref_aspect_ratio = ref_img_square.width / ref_img_square.height
        print(f"🔄 Expanding generated image canvas to match reference aspect ratio ({ref_aspect_ratio:.3f})...")
        print(f"  Generated image size before expansion: {generated_image.size}")
        generated_image = expand_canvas_to_aspect_ratio(generated_image, ref_aspect_ratio)
        print(f"  Generated image size after expansion: {generated_image.size}")
        
        # Save generated image temporarily for debugging (auto-deleted)
        import tempfile
        import os
        
        # We use a try-finally block or context manager logic implicitly by not relying on the file logic downstream.
        # But to be explicit and use the file path only if needed:
        
        # Create a temp file
        # Note: In Windows, you can't open a file twice, so we just save it.
        # But here we just want to save it and let it be deleted later or just print path.
        # Since we don't READ it back from path in this function (we use generated_image object),
        # we can just use NamedTemporaryFile as a context manager if we want it to exist during extraction?
        # Actually extraction doesn't use the file. So the file is purely for the USER to verify if they are debugging.
        # If we auto-delete it immediately, the user can't see it.
        # But the user asked to "delete once it has been used".
        # It is "used" by the extraction process (in memory).
        # So effectively, we save it, print path, and it gets deleted on exit.
        
        # Save generated image PERSISTENTLY for manual verification
        import os
        from datetime import datetime
        
        # Create directory if it doesn't exist
        save_dir = os.path.join(os.path.dirname(__file__), "..", "generated_images")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"generated_{timestamp}.png")
        generated_image.save(save_path)
        print(f"💾 Saved generated image to: {save_path}")
        
        # Extract coordinates from generated image ONLY (like test_extraction.py)
        print("🔍 Extracting coordinates from generated image...")
        extracted_lines = self.extractor.extract_lines_from_generated_image_only(
            generated_image,
            expected_line_count=expected_line_count,
            reference_image=ref_img_square if settings.USE_REFERENCE_BASED_MATCHING else None
        )
        
        if not extracted_lines or len(extracted_lines) == 0:
            raise RuntimeError(
                "Failed to extract lines from generated image. "
                "The image may not contain visible measurement lines."
            )
        
        print(f"✓ Extracted {len(extracted_lines)} lines from generated image")
        
        # NEW: Extract labels using OCR if reference_lines were provided
        if reference_lines and len(reference_lines) > 0:
            print("\n🏷️  Extracting labels from generated image using OCR...")
            
            # Get expected labels from reference lines
            expected_labels = [line.get('label', '') for line in reference_lines]
            
            # Extract labels using OCR
            ocr_labels = self.ocr.extract_labels(generated_image, expected_labels=expected_labels)
            
            # Match extracted lines to OCR labels
            extracted_lines = self.ocr.match_lines_to_labels(
                extracted_lines,
                ocr_labels,
                reference_lines
            )
            
            print(f"✓ Matched lines to labels via OCR")
        
        lines = self._convert_to_line_drawings(extracted_lines)

        
        # Debug: Log final lines returned to frontend
        print(f"\n📊 FINAL LINES RETURNED TO FRONTEND ({len(lines)} total):")
        import json
        import os
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: start=({line.start.x:.6f}, {line.start.y:.6f}), "
                  f"end=({line.end.x:.6f}, {line.end.y:.6f}), "
                  f"label='{line.label}'")
            # #region agent log
            log_data = {
                "location": "line_detector.py:162",
                "message": "Line returned to frontend",
                "data": {
                    "lineIndex": i,
                    "startX": float(line.start.x),
                    "startY": float(line.start.y),
                    "endX": float(line.end.x),
                    "endY": float(line.end.y),
                    "label": line.label or ""
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D"
            }
            try:
                with open("/Users/ayush/Downloads/spatial_understanding-main/.cursor/debug.log", "a") as f:
                    f.write(json.dumps(log_data) + "\n")
            except Exception:
                pass
            # #endregion
        
        method_used = f"image_generation_{model_name}"
        
        # Calculate confidence using generated image (not input image)
        confidence = self._calculate_confidence(lines, generated_image)
        
        # Convert generated image to base64 for frontend
        generated_image_base64 = encode_image_to_base64(generated_image)
        generated_image_base64 = f"data:image/png;base64,{generated_image_base64}"
        
        # Convert resized input image to base64 for frontend (clean image without black lines)
        input_image_base64_resized = encode_image_to_base64(input_img_square)
        input_image_base64_resized = f"data:image/png;base64,{input_image_base64_resized}"
        
        processing_time = time.time() - start_time
        
        return lines, method_used, confidence, generated_image_base64, input_image_base64_resized
    
    def _convert_to_line_drawings(self, lines: List[dict]) -> List[LineDrawing]:
        """Convert dict lines to LineDrawing objects"""
        print(f"\n🔄 Converting {len(lines)} lines to LineDrawing objects...")
        result = []
        for i, line in enumerate(lines):
            # Handle different input formats
            start = line.get("start", {})
            end = line.get("end", {})
            
            if isinstance(start, list):
                start = {"x": start[1] / 1000, "y": start[0] / 1000}
            if isinstance(end, list):
                end = {"x": end[1] / 1000, "y": end[0] / 1000}
            
            start_x = start.get("x", 0)
            start_y = start.get("y", 0)
            end_x = end.get("x", 0)
            end_y = end.get("y", 0)
            
            # Check for invalid values
            if start_x is None or start_y is None or end_x is None or end_y is None:
                print(f"  ❌ Line {i+1} REJECTED: None values in coordinates")
                continue
            
            # Check bounds
            coords_valid = (0 <= start_x <= 1 and 0 <= start_y <= 1 and 0 <= end_x <= 1 and 0 <= end_y <= 1)
            if not coords_valid:
                print(f"  ⚠️  Line {i+1} has coordinates outside 0-1 range:")
                print(f"      start: ({start_x:.6f}, {start_y:.6f}), end: ({end_x:.6f}, {end_y:.6f})")
            
            line_drawing = LineDrawing(
                start=Point(x=start_x, y=start_y),
                end=Point(x=end_x, y=end_y),
                label=line.get("label", ""),
                confidence=line.get("confidence")
            )
            
            result.append(line_drawing)
            print(f"  ✓ Line {i+1} converted: start=({start_x:.6f}, {start_y:.6f}), end=({end_x:.6f}, {end_y:.6f})")
        
        print(f"✓ Converted {len(result)}/{len(lines)} lines successfully")
        return result
    
    def _refine_all_lines(
        self,
        lines: List[dict],
        image: Image.Image
    ) -> List[dict]:
        """Refine all lines by snapping to edges"""
        img_arr = pil_to_numpy(image)
        edges = detect_edges(img_arr)
        h, w = img_arr.shape[:2]
        
        refined = []
        for line in lines:
            start = line.get("start", {})
            end = line.get("end", {})
            
            # Convert to pixel coordinates
            start_pixel = (
                int(start.get("x", 0) * w),
                int(start.get("y", 0) * h)
            )
            end_pixel = (
                int(end.get("x", 0) * w),
                int(end.get("y", 0) * h)
            )
            
            # Snap to edges
            refined_start = snap_point_to_edge(start_pixel, edges)
            refined_end = snap_point_to_edge(end_pixel, edges)
            
            refined.append({
                "start": {"x": refined_start[0] / w, "y": refined_start[1] / h},
                "end": {"x": refined_end[0] / w, "y": refined_end[1] / h},
                "label": line.get("label", "")
            })
        
        return refined
    
    def _calculate_confidence(
        self,
        lines: List[LineDrawing],
        image: Image.Image
    ) -> float:
        """Calculate confidence score based on edge alignment"""
        if not lines:
            return 0.0
        
        img_arr = pil_to_numpy(image)
        edges = detect_edges(img_arr)
        h, w = img_arr.shape[:2]
        
        total_score = 0.0
        for line in lines:
            # Check if endpoints are near edges
            start_pixel = (int(line.start.x * w), int(line.start.y * h))
            end_pixel = (int(line.end.x * w), int(line.end.y * h))
            
            # Count edge pixels near the line
            # Sample points along the line
            num_samples = 10
            edge_count = 0
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int(start_pixel[0] * (1 - t) + end_pixel[0] * t)
                y = int(start_pixel[1] * (1 - t) + end_pixel[1] * t)
                
                if 0 <= y < h and 0 <= x < w:
                    # Check small neighborhood
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if edges[ny, nx] > 0:
                                    edge_count += 1
                                    break
                        if edges[y + dy, x] > 0:
                            break
            
            # Score based on edge alignment
            score = min(1.0, edge_count / (num_samples * 5))
            total_score += score
        
        return total_score / len(lines) if lines else 0.0

