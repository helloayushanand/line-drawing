import time
from typing import List, Tuple, Optional
from PIL import Image
from services.gemini_service import GeminiService
from services.flux_service import FluxService
from services.coordinate_extractor import CoordinateExtractor
from services.ocr_service import OCRService
from utils.image_utils import decode_base64_image, pil_to_numpy, encode_image_to_base64, resize_image, expand_canvas_to_aspect_ratio
from utils.edge_detection import detect_edges
from models.schemas import LineDrawing, Point
from config import settings
import numpy as np

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
        self.ocr = OCRService()
    
    def detect_lines(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        product_type: str = None,
        expected_line_count: int = None,
        line_types: List[str] = None,
        model: str = "gemini",
        reference_lines: List[dict] = None,
    ) -> Tuple[List[LineDrawing], str, float, Optional[str], Optional[str]]:
        """Detect lines using image generation"""
        start_time = time.time()
        
        input_img = decode_base64_image(input_image_base64)
        ref_img = decode_base64_image(reference_image_base64)
        
        print("🔄 Resizing images to max 1024px for consistent processing...")
        input_img_square = resize_image(input_img, 1024)
        ref_img_square = resize_image(ref_img, 1024)
        print(f"✓ Input image resized: {input_img.size} -> {input_img_square.size}")
        print(f"✓ Reference image resized: {ref_img.size} -> {ref_img_square.size}")
        
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
            generated_image = self.flux.generate_image_with_lines(
                input_image_base64_square,
                reference_image_base64_square,
                product_type=product_type,
                line_types=line_types,
                expected_line_count=expected_line_count,
                reference_lines=reference_lines
            )

        else:
            print("\n📸 Attempting Gemini 3 image generation...")
            generated_image = self.gemini.generate_image_with_lines(
                input_image_base64_square,
                reference_image_base64_square,
                product_type,
                line_types,
                expected_line_count=expected_line_count,
                reference_lines=reference_lines
            )
        
        if not generated_image:
            raise RuntimeError(
                f"{model_name} image generation failed. No image was generated. "
                "Please check your API key and that the selected model supports image generation."
            )
        
        if max(generated_image.size) > 1024:
            print(f"⚠️  Generated image size is {generated_image.size}, resizing to max 1024...")
            generated_image = resize_image(generated_image, 1024)
        
        ref_aspect_ratio = ref_img_square.width / ref_img_square.height
        print(f"🔄 Expanding generated image canvas to match reference aspect ratio ({ref_aspect_ratio:.3f})...")
        print(f"  Generated image size before expansion: {generated_image.size}")
        generated_image = expand_canvas_to_aspect_ratio(generated_image, ref_aspect_ratio)
        print(f"  Generated image size after expansion: {generated_image.size}")
        
        import os
        from datetime import datetime
        
        save_dir = os.path.join(os.path.dirname(__file__), "..", "generated_images")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"generated_{timestamp}.png")
        generated_image.save(save_path)
        print(f"💾 Saved generated image to: {save_path}")
        
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
        
        if reference_lines and len(reference_lines) > 0:
            print("\n🏷️  Extracting labels from generated image using OCR...")
            
            expected_labels = [line.get('label', '') for line in reference_lines]
            ocr_labels = self.ocr.extract_labels(generated_image, expected_labels=expected_labels)
            extracted_lines = self.ocr.match_lines_to_labels(
                extracted_lines,
                ocr_labels,
                reference_lines
            )
            
            print(f"✓ Matched lines to labels via OCR")
        
        lines = self._convert_to_line_drawings(extracted_lines)
        
        print(f"\n📊 FINAL LINES RETURNED TO FRONTEND ({len(lines)} total):")
        import json
        import os
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: start=({line.start.x:.6f}, {line.start.y:.6f}), "
                  f"end=({line.end.x:.6f}, {line.end.y:.6f}), "
                  f"label='{line.label}'")
        
        method_used = f"image_generation_{model_name}"
        confidence = self._calculate_confidence(lines, generated_image)
        
        generated_image_base64 = encode_image_to_base64(generated_image)
        generated_image_base64 = f"data:image/png;base64,{generated_image_base64}"
        input_image_base64_resized = encode_image_to_base64(input_img_square)
        input_image_base64_resized = f"data:image/png;base64,{input_image_base64_resized}"
        
        processing_time = time.time() - start_time
        
        return lines, method_used, confidence, generated_image_base64, input_image_base64_resized
    
    def _convert_to_line_drawings(self, lines: List[dict]) -> List[LineDrawing]:
        """Convert dict lines to LineDrawing objects"""
        print(f"\n🔄 Converting {len(lines)} lines to LineDrawing objects...")
        result = []
        for i, line in enumerate(lines):
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
            
            if start_x is None or start_y is None or end_x is None or end_y is None:
                print(f"  ❌ Line {i+1} REJECTED: None values in coordinates")
                continue
            
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
            start_pixel = (int(line.start.x * w), int(line.start.y * h))
            end_pixel = (int(line.end.x * w), int(line.end.y * h))
            
            num_samples = 10
            edge_count = 0
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int(start_pixel[0] * (1 - t) + end_pixel[0] * t)
                y = int(start_pixel[1] * (1 - t) + end_pixel[1] * t)
                
                if 0 <= y < h and 0 <= x < w:
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if edges[ny, nx] > 0:
                                    edge_count += 1
                                    break
                        if edges[y + dy, x] > 0:
                            break
            
            score = min(1.0, edge_count / (num_samples * 5))
            total_score += score
        
        return total_score / len(lines) if lines else 0.0

