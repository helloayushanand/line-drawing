import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from utils.edge_detection import detect_lines_hough, detect_lines_by_color, detect_edges
from utils.image_utils import pil_to_numpy
from config import settings

class CoordinateExtractor:
    def __init__(self):
        pass
    
    def extract_lines_from_generated_image_only(
        self,
        generated_image: Image.Image,
        expected_line_count: Optional[int] = None,
        reference_image: Optional[Image.Image] = None
    ) -> List[dict]:
        """Extract line coordinates from generated image"""
        try:
            if settings.USE_SIMPLE_HOUGH:
                print("  📐 Using simple Hough transform detection...")
                cv_lines = self.extract_lines_simple_hough(
                    generated_image
                )
                print(f"  ✓ Simple Hough found {len(cv_lines)} lines")
            else:
                product_mask = self._detect_product_mask(generated_image)
                print(f"  ✓ Product mask detected: {product_mask.shape}")
                cv_lines = self._extract_via_cv(generated_image, product_mask)
                print(f"  ✓ CV detection found {len(cv_lines)} lines")
            
            if 'product_mask' not in locals():
                product_mask = self._detect_product_mask(generated_image)
                print(f"  ✓ Product mask detected (for filtering): {product_mask.shape}")
            
            filtered_lines = self._filter_measurement_lines(cv_lines, product_mask, generated_image)
            print(f"  ✓ After filtering: {len(filtered_lines)} measurement lines detected")
            
            if len(filtered_lines) == 0 and len(cv_lines) > 0 and settings.USE_SIMPLE_HOUGH:
                 print("  ⚠️  Filtering removed all lines. Using raw candidates for selection.")
                 filtered_lines = cv_lines

            if expected_line_count and len(filtered_lines) > expected_line_count:
                if settings.USE_REFERENCE_BASED_MATCHING and reference_image is not None:
                    print("  🔄 Using reference-based matching...")
                    filtered_lines = self._rank_and_select_lines_reference_based(
                        filtered_lines, expected_line_count, generated_image, product_mask, reference_image
                    )
                    print(f"  ✓ After reference-based ranking: {len(filtered_lines)} distinct lines selected")
                else:
                    print("  🔄 Using current ranking logic...")
                    filtered_lines = self._rank_and_select_lines(filtered_lines, expected_line_count, generated_image, product_mask)
                    print(f"  ✓ After ranking: {len(filtered_lines)} distinct lines selected")
            
            return filtered_lines
        except Exception as e:
            print(f"❌ Error in extract_lines_from_generated_image_only: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_lines_simple_hough(
        self,
        image: Image.Image,
        black_threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
        hough_threshold: int = 50
    ) -> List[dict]:
        """Extract black lines using Hough transform"""
        img_arr = pil_to_numpy(image)
        h, w = img_arr.shape[:2]
        
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result.append({
                    "start": {"x": x1 / w, "y": y1 / h},
                    "end": {"x": x2 / w, "y": y2 / h},
                    "label": ""
                })
        
                result.append({
                    "start": {"x": x1 / w, "y": y1 / h},
                    "end": {"x": x2 / w, "y": y2 / h},
                    "label": ""
                })
        
        return result

    def _extract_via_cv(self, image: Image.Image, product_mask: Optional[np.ndarray] = None) -> List[dict]:
        """Extract lines using color-based or edge-based detection"""
        img_arr = pil_to_numpy(image)
        
        try:
            lines = detect_lines_by_color(img_arr, product_mask)
        except Exception as e:
            print(f"  ⚠️  Color-based detection failed: {e}, falling back to edge-based")
            lines = detect_lines_hough(img_arr)
        
        h, w = img_arr.shape[:2]
        normalized_lines = []
        for x1, y1, x2, y2 in lines:
            normalized_lines.append({
                "start": {"x": x1 / w, "y": y1 / h},
                "end": {"x": x2 / w, "y": y2 / h},
                "label": ""
            })
        
        return normalized_lines

    def _detect_product_mask(self, original_image: Image.Image) -> np.ndarray:
        """Detect product mask from white background"""
        img_arr = pil_to_numpy(original_image)
        h, w = img_arr.shape[:2]
        
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros((h, w), dtype=np.uint8)
        
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _filter_measurement_lines(
        self,
        lines: List[dict],
        product_mask: np.ndarray,
        generated_image: Image.Image
    ) -> List[dict]:
        """Filter measurement lines by length, contrast, and position"""
        h, w = product_mask.shape[:2]
        img_arr = pil_to_numpy(generated_image)
        
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        filtered_lines = []
        mask_for_dist = product_mask.astype(np.uint8)
        if len(mask_for_dist.shape) > 2:
            mask_for_dist = cv2.cvtColor(mask_for_dist, cv2.COLOR_RGB2GRAY)
        
        inverted_mask = cv2.bitwise_not(mask_for_dist)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        min_length = min(w, h) * settings.MEASUREMENT_LINE_MIN_LENGTH_RATIO
        
        for line in lines:
            start_x = int(line["start"]["x"] * w)
            start_y = int(line["start"]["y"] * h)
            end_x = int(line["end"]["x"] * w)
            end_y = int(line["end"]["y"] * h)
            
            start_x = max(0, min(w - 1, start_x))
            start_y = max(0, min(h - 1, start_y))
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            line_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            if line_length < min_length:
                continue
            
            num_samples = 10
            inside_count = 0
            contrast_values = []
            min_distance_to_edge = float('inf')
            
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int(start_x * (1 - t) + end_x * t)
                y = int(start_y * (1 - t) + end_y * t)
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                if 0 <= y < h and 0 <= x < w:
                    if product_mask[y, x] > 0:
                        inside_count += 1
                    
                    dist_to_edge = dist_transform[y, x]
                    min_distance_to_edge = min(min_distance_to_edge, dist_to_edge)
                    pixel_value = gray[y, x]
                    contrast_values.append(pixel_value)
            
            if len(contrast_values) == 0:
                continue
            
            inside_ratio = inside_count / (num_samples + 1)
            outside_ratio = 1.0 - inside_ratio
            avg_contrast = np.mean(contrast_values)
            max_contrast = np.max(contrast_values)
            
            is_high_contrast = (avg_contrast < settings.MEASUREMENT_LINE_CONTRAST_THRESHOLD and 
                              max_contrast < settings.MEASUREMENT_LINE_CONTRAST_MAX)
            
            if not is_high_contrast:
                continue
            
            is_clear_position = (inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD or 
                               outside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD)
            
            if not is_clear_position:
                continue
            
            if outside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD:
                if min_distance_to_edge < settings.MEASUREMENT_LINE_EDGE_SEPARATION:
                    continue
            
            filtered_lines.append(line)
        
        return filtered_lines
    
    def _are_lines_similar(
        self,
        line1: dict,
        line2: dict,
        image: Image.Image
    ) -> bool:
        """
        Check if two lines are too similar (overlapping/duplicate)
        Returns True if lines are considered similar/duplicate
        """
        h, w = image.size[1], image.size[0]  # PIL Image size is (width, height)
        
        # Convert to pixel coordinates
        line1_start_x = line1["start"]["x"] * w
        line1_start_y = line1["start"]["y"] * h
        line1_end_x = line1["end"]["x"] * w
        line1_end_y = line1["end"]["y"] * h
        
        line2_start_x = line2["start"]["x"] * w
        line2_start_y = line2["start"]["y"] * h
        line2_end_x = line2["end"]["x"] * w
        line2_end_y = line2["end"]["y"] * h
        
        # Calculate midpoints
        line1_mid_x = (line1_start_x + line1_end_x) / 2
        line1_mid_y = (line1_start_y + line1_end_y) / 2
        line2_mid_x = (line2_start_x + line2_end_x) / 2
        line2_mid_y = (line2_start_y + line2_end_y) / 2
        
        # Calculate distance between midpoints (normalized)
        max_distance = np.sqrt(w**2 + h**2)
        midpoint_distance = np.sqrt(
            (line1_mid_x - line2_mid_x)**2 + (line1_mid_y - line2_mid_y)**2
        )
        normalized_distance = midpoint_distance / max_distance if max_distance > 0 else 0
        
        # Check distance threshold
        if normalized_distance < settings.LINE_SIMILARITY_DISTANCE_THRESHOLD:
            return True
        
        # Calculate angles of both lines
        line1_dx = line1_end_x - line1_start_x
        line1_dy = line1_end_y - line1_start_y
        line1_angle = np.degrees(np.arctan2(line1_dy, line1_dx))
        
        line2_dx = line2_end_x - line2_start_x
        line2_dy = line2_end_y - line2_start_y
        line2_angle = np.degrees(np.arctan2(line2_dy, line2_dx))
        
        # Normalize angles to 0-180 range (lines are undirected)
        line1_angle = abs(line1_angle) % 180
        line2_angle = abs(line2_angle) % 180
        
        # Calculate angle difference (handle wrap-around)
        angle_diff = abs(line1_angle - line2_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # Check if lines are parallel (similar angle) and close
        if angle_diff < settings.LINE_SIMILARITY_ANGLE_THRESHOLD:
            # If lines are parallel and close, check overlap
            # Calculate perpendicular distance between parallel lines
            # For parallel lines: distance = |ax + by + c| / sqrt(a^2 + b^2)
            # Using line equation: ax + by + c = 0
            if abs(line1_dx) < 1e-6:  # Vertical line
                perp_distance = abs(line1_mid_x - line2_mid_x) / w
            elif abs(line1_dy) < 1e-6:  # Horizontal line
                perp_distance = abs(line1_mid_y - line2_mid_y) / h
            else:
                # General line: y = mx + b
                m1 = line1_dy / line1_dx
                b1 = line1_start_y - m1 * line1_start_x
                m2 = line2_dy / line2_dx
                b2 = line2_start_y - m2 * line2_start_x
                
                # Distance between parallel lines
                perp_distance = abs(b1 - b2) / np.sqrt(m1**2 + 1) / max(w, h)
            
            # If parallel and very close, they're similar
            if perp_distance < settings.LINE_SIMILARITY_DISTANCE_THRESHOLD:
                return True
        
        return False
    
    def _rank_and_select_lines(
        self,
        lines: List[dict],
        expected_count: int,
        original_image: Image.Image,
        product_mask: np.ndarray,
        existing_lines: List[dict] = None
    ) -> List[dict]:

        """
        Rank lines and select top N based on:
        - Line length (longer = more likely measurement line)
        - Distance from product mask/contour (further = more likely measurement line)
        """
        if len(lines) <= expected_count:
            return lines
        
        h, w = product_mask.shape[:2]
        scored_lines = []
        
        # Pre-calculate distance transform once for all lines (efficiency)
        # Convert mask to uint8 if needed and ensure it's single channel
        mask_for_dist = product_mask.astype(np.uint8)
        if len(mask_for_dist.shape) > 2:
            mask_for_dist = cv2.cvtColor(mask_for_dist, cv2.COLOR_RGB2GRAY)
        elif len(mask_for_dist.shape) == 1:
            # If 1D, reshape to 2D
            mask_for_dist = mask_for_dist.reshape((h, w))
        
        # Ensure mask is 2D
        if len(mask_for_dist.shape) != 2:
            raise ValueError(f"Invalid mask shape: {mask_for_dist.shape}, expected 2D")
        
        # Invert mask: 0 = product, 255 = background
        inverted_mask = cv2.bitwise_not(mask_for_dist)
        
        # Calculate distance transform once for all lines
        # distanceTransform requires single channel uint8 image
        dist_transform = cv2.distanceTransform(
            inverted_mask,
            cv2.DIST_L2,
            5
        )
        
        for line in lines:
            # Calculate line length
            start_x = line["start"]["x"] * w
            start_y = line["start"]["y"] * h
            end_x = line["end"]["x"] * w
            end_y = line["end"]["y"] * h
            
            line_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # Calculate average distance from product mask
            # Sample points along line and find min distance to product edge
            num_samples = 10
            min_distances = []
            
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int(start_x * (1 - t) + end_x * t)
                y = int(start_y * (1 - t) + end_y * t)
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                if 0 <= y < h and 0 <= x < w:
                    min_distances.append(dist_transform[y, x])
            
            avg_distance = np.mean(min_distances) if min_distances else 0
            
            max_length = np.sqrt(w**2 + h**2)
            normalized_length = line_length / max_length if max_length > 0 else 0
            max_distance = min(w, h) / 2
            normalized_distance = avg_distance / max_distance if max_distance > 0 else 0
            score = 0.6 * normalized_length + 0.4 * normalized_distance
            
            scored_lines.append({
                "line": line,
                "score": score,
                "length": line_length,
                "distance": avg_distance
            })
        
        scored_lines.sort(key=lambda x: x["score"], reverse=True)
        
        selected = []
        for item in scored_lines:
            if len(selected) >= expected_count:
                break
            
            candidate_line = item["line"]
            is_distinct = True
            
            for selected_line in selected:
                if self._are_lines_similar(candidate_line, selected_line, original_image):
                    is_distinct = False
                    break
            
            if is_distinct and existing_lines:
                for existing_line in existing_lines:
                    if self._are_lines_similar(candidate_line, existing_line, original_image):
                        is_distinct = False
                        break
            
            if is_distinct:
                selected.append(candidate_line)
        
        return selected
    
    def _rank_and_select_lines_reference_based(
        self,
        lines: List[dict],
        expected_count: int,
        generated_image: Image.Image,
        product_mask: np.ndarray,
        reference_image: Image.Image
    ) -> List[dict]:
        """Rank and select lines by matching to reference image pattern"""
        if len(lines) <= expected_count:
            return lines
        
        print("    📐 Extracting lines from reference image...")
        ref_product_mask = self._detect_product_mask(reference_image)
        ref_cv_lines = self._extract_via_cv(reference_image, ref_product_mask)
        ref_filtered_lines = self._filter_measurement_lines(ref_cv_lines, ref_product_mask, reference_image)
        
        if not ref_filtered_lines:
            print("    ⚠️  No lines found in reference image, falling back to current logic")
            return self._rank_and_select_lines(lines, expected_count, generated_image, product_mask)
        
        print(f"    ✓ Found {len(ref_filtered_lines)} reference lines")
        
        ref_lines_classified = []
        for ref_line in ref_filtered_lines:
            inside_ratio = self._calculate_inside_ratio(ref_line, ref_product_mask, reference_image)
            is_inside = inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD
            ref_lines_classified.append({
                "line": ref_line,
                "is_inside": is_inside,
                "inside_ratio": inside_ratio
            })
        
        ref_inside_count = sum(1 for rl in ref_lines_classified if rl["is_inside"])
        ref_outside_count = len(ref_lines_classified) - ref_inside_count
        print(f"    📊 Reference pattern: {ref_inside_count} inside, {ref_outside_count} outside")
        
        selected = []
        used_generated_indices = set()
        
        h, w = product_mask.shape[:2]
        gen_h, gen_w = generated_image.size[1], generated_image.size[0]
        
        # Pre-calculate distance transform for generated image
        mask_for_dist = product_mask.astype(np.uint8)
        if len(mask_for_dist.shape) > 2:
            mask_for_dist = cv2.cvtColor(mask_for_dist, cv2.COLOR_RGB2GRAY)
        inverted_mask = cv2.bitwise_not(mask_for_dist)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        
        for ref_item in ref_lines_classified:
            if len(selected) >= expected_count:
                break
            
            ref_line = ref_item["line"]
            ref_is_inside = ref_item["is_inside"]
            
            best_match = None
            best_score = -1
            best_idx = -1
            
            for gen_idx, gen_line in enumerate(lines):
                if gen_idx in used_generated_indices:
                    continue
                
                gen_inside_ratio = self._calculate_inside_ratio(gen_line, product_mask, generated_image)
                gen_is_inside = gen_inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD
                
                if gen_is_inside != ref_is_inside:
                    continue
                
                score = self._calculate_line_match_score(
                    ref_line, gen_line,
                    ref_product_mask, product_mask,
                    reference_image, generated_image,
                    dist_transform, h, w
                )
                
                if score > best_score:
                    best_score = score
                    best_match = gen_line
                    best_idx = gen_idx
            
            if best_match is not None and best_score > 0:
                is_duplicate = False
                for already_selected in selected:
                    if self._are_lines_similar(best_match, already_selected, generated_image):
                        is_duplicate = True
                        print(f"    ⚠️  Skipping duplicate line (too similar to already selected line)")
                        break
                
                if not is_duplicate:
                    selected.append(best_match)
                    used_generated_indices.add(best_idx)
                    print(f"    ✓ Matched reference line: score={best_score:.3f}, inside={ref_is_inside}")
            else:
                print(f"    ⚠️  No match found for reference line (inside={ref_is_inside})")
        
        if len(selected) < expected_count:
            print(f"    ⚠️  Only found {len(selected)} matches, filling remaining {expected_count - len(selected)} slots with current logic")
            remaining_lines = [line for idx, line in enumerate(lines) if idx not in used_generated_indices]
            if remaining_lines:
                remaining_selected = self._rank_and_select_lines(
                    remaining_lines, expected_count - len(selected), generated_image, product_mask,
                    existing_lines=selected
                )
                selected.extend(remaining_selected)
        
        return selected[:expected_count]
    
    def _calculate_inside_ratio(self, line: dict, product_mask: np.ndarray, image: Image.Image) -> float:
        """Calculate what ratio of the line is inside the product"""
        h, w = product_mask.shape[:2]
        start_x = int(line["start"]["x"] * w)
        start_y = int(line["start"]["y"] * h)
        end_x = int(line["end"]["x"] * w)
        end_y = int(line["end"]["y"] * h)
        
        num_samples = 10
        inside_count = 0
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(start_x * (1 - t) + end_x * t)
            y = int(start_y * (1 - t) + end_y * t)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            if 0 <= y < h and 0 <= x < w:
                if product_mask[y, x] > 0:
                    inside_count += 1
        
        return inside_count / (num_samples + 1)
    
    def _calculate_line_match_score(
        self,
        ref_line: dict,
        gen_line: dict,
        ref_product_mask: np.ndarray,
        gen_product_mask: np.ndarray,
        ref_image: Image.Image,
        gen_image: Image.Image,
        gen_dist_transform: np.ndarray,
        h: int,
        w: int
    ) -> float:
        """
        Calculate match score between reference and generated line.
        Score components:
        - 0.4 * length_score (normalized length)
        - 0.3 * distance_score (distance from product edge)
        - 0.2 * angle_similarity (soft, no rejection)
        - 0.1 * position_bonus (if reference has outside lines, bonus for outside candidates)
        """
        # Calculate length score
        ref_start_x = ref_line["start"]["x"] * ref_image.size[0]
        ref_start_y = ref_line["start"]["y"] * ref_image.size[1]
        ref_end_x = ref_line["end"]["x"] * ref_image.size[0]
        ref_end_y = ref_line["end"]["y"] * ref_image.size[1]
        ref_length = np.sqrt((ref_end_x - ref_start_x)**2 + (ref_end_y - ref_start_y)**2)
        
        gen_start_x = gen_line["start"]["x"] * w
        gen_start_y = gen_line["start"]["y"] * h
        gen_end_x = gen_line["end"]["x"] * w
        gen_end_y = gen_line["end"]["y"] * h
        gen_length = np.sqrt((gen_end_x - gen_start_x)**2 + (gen_end_y - gen_start_y)**2)
        
        # Normalize lengths
        ref_max_length = np.sqrt(ref_image.size[0]**2 + ref_image.size[1]**2)
        gen_max_length = np.sqrt(w**2 + h**2)
        ref_length_norm = ref_length / ref_max_length if ref_max_length > 0 else 0
        gen_length_norm = gen_length / gen_max_length if gen_max_length > 0 else 0
        
        # Length similarity
        length_diff = abs(ref_length_norm - gen_length_norm)
        length_score = 1.0 - min(length_diff / 0.5, 1.0)  # 0.5 = 50% difference threshold
        
        # Distance score (for generated line)
        num_samples = 10
        min_distances = []
        for i in range(num_samples + 1):
            t = i / num_samples
            x = int(gen_start_x * (1 - t) + gen_end_x * t)
            y = int(gen_start_y * (1 - t) + gen_end_y * t)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            if 0 <= y < h and 0 <= x < w:
                min_distances.append(gen_dist_transform[y, x])
        
        avg_distance = np.mean(min_distances) if min_distances else 0
        max_distance = min(w, h) / 2
        distance_score = avg_distance / max_distance if max_distance > 0 else 0
        
        # Angle similarity (soft, relative to product)
        ref_angle = np.degrees(np.arctan2(ref_end_y - ref_start_y, ref_end_x - ref_start_x))
        gen_angle = np.degrees(np.arctan2(gen_end_y - gen_start_y, gen_end_x - gen_start_x))
        
        # Normalize angles to 0-180 range
        ref_angle = abs(ref_angle) % 180
        gen_angle = abs(gen_angle) % 180
        
        # Calculate angle difference
        angle_diff = abs(ref_angle - gen_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # Angle similarity (0.0 to 1.0)
        angle_score = 1.0 - (angle_diff / 90.0)
        
        # Position bonus (small bonus for outside lines if reference has outside lines)
        gen_inside_ratio = self._calculate_inside_ratio(gen_line, gen_product_mask, gen_image)
        gen_is_outside = gen_inside_ratio < (1.0 - settings.MEASUREMENT_LINE_POSITION_THRESHOLD)
        position_bonus = 0.1 if gen_is_outside else 0.0
        
        # Combined score
        final_score = (
            0.4 * length_score +
            0.3 * distance_score +
            0.2 * angle_score +
            0.1 * position_bonus
        )
        
        return final_score

