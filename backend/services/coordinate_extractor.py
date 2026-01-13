import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from utils.edge_detection import detect_lines_hough, detect_lines_by_color, snap_point_to_edge, detect_edges
from utils.image_utils import pil_to_numpy
import io
import json
from config import settings

class CoordinateExtractor:
    def __init__(self):
        # Genai client not needed - using CV-only extraction
        pass
    
    # [DEPRECATED] extract_lines_from_generated_image
    # This method was removed because we now strictly use extract_lines_from_generated_image_only
    # See coordinate_extractor_backup.py if you need the old logic (image filtering, diffing, etc.)

    
    def extract_lines_from_generated_image_only(
        self,
        generated_image: Image.Image,
        expected_line_count: Optional[int] = None,
        reference_image: Optional[Image.Image] = None
    ) -> List[dict]:
        """
        Extract line coordinates from generated image only (no original needed)
        Uses CV detection directly on generated image
        """
        try:
            # Check if simple Hough should be used for DETECTION
            if settings.USE_SIMPLE_HOUGH:
                print("  📐 Using simple Hough transform detection...")
                cv_lines = self.extract_lines_simple_hough(
                    generated_image,
                    black_threshold=settings.BLACK_LINE_THRESHOLD,
                    min_line_length=settings.MIN_LINE_LENGTH,
                    max_line_gap=settings.MAX_LINE_GAP,
                    hough_threshold=settings.HOUGH_LINE_THRESHOLD
                )
                print(f"  ✓ Simple Hough found {len(cv_lines)} lines")
            else:
                # Detect product mask first (needed for color-based detection)
                product_mask = self._detect_product_mask(generated_image)
                print(f"  ✓ Product mask detected: {product_mask.shape}")
                
                # Direct CV detection on generated image (with product mask for color-based detection)
                cv_lines = self._extract_via_cv(generated_image, product_mask)
                print(f"  ✓ CV detection found {len(cv_lines)} lines")
            
            # Extend detected lines to full extent
            print("  🔄 Extending lines to full extent...")
            img_arr = pil_to_numpy(generated_image)
            h_ext, w_ext = img_arr.shape[:2]
            if len(img_arr.shape) == 3:
                gray_ext = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            else:
                gray_ext = img_arr
            _, binary_mask = cv2.threshold(gray_ext, settings.BLACK_LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            
            extended_lines = []
            for line in cv_lines:
                try:
                    extended_line = self._extend_line_to_full_extent(line, generated_image, binary_mask)
                    extended_lines.append(extended_line)
                except Exception as e:
                    print(f"  ⚠️  Failed to extend line: {e}, using original")
                    extended_lines.append(line)
            cv_lines = extended_lines
            print(f"  ✓ Extended {len(cv_lines)} lines")
            
            # Detect product mask if not already detected (needed for filtering/selection)
            if 'product_mask' not in locals():
                product_mask = self._detect_product_mask(generated_image)
                print(f"  ✓ Product mask detected (for filtering): {product_mask.shape}")
            
            # Store raw lines before filtering (for fallback)
            raw_cv_lines = cv_lines.copy()
            
            # Filter to keep only measurement lines (both inside and outside product)
            # Note: Color-based detection already excludes product edges, but we still filter
            # to ensure lines meet length and position requirements
            filtered_lines = self._filter_measurement_lines(cv_lines, product_mask, generated_image)
            print(f"  ✓ After strict filtering: {len(filtered_lines)} measurement lines detected")
            
            # CRITICAL: Ensure we have enough lines to match expected count
            if expected_line_count and len(filtered_lines) < expected_line_count:
                print(f"  ⚠️  Only {len(filtered_lines)} lines after filtering, but {expected_line_count} expected")
                print(f"  🔄 Attempting relaxed filtering to recover more lines...")
                
                # Try relaxed filtering
                relaxed_filtered = self._filter_measurement_lines_relaxed(raw_cv_lines, product_mask, generated_image)
                print(f"  ✓ After relaxed filtering: {len(relaxed_filtered)} lines detected")
                
                # Use relaxed filtering result if it gives us more lines
                if len(relaxed_filtered) >= len(filtered_lines):
                    filtered_lines = relaxed_filtered
                    print(f"  ✓ Using relaxed filtering result ({len(filtered_lines)} lines)")
                
                # If still insufficient, fall back to raw lines (but still apply ranking)
                if len(filtered_lines) < expected_line_count:
                    print(f"  ⚠️  Still insufficient lines ({len(filtered_lines)} < {expected_line_count})")
                    print(f"  🔄 Falling back to raw detected lines and applying ranking...")
                    # Use all raw lines, but we'll rank them to get the best ones
                    filtered_lines = raw_cv_lines
                else:
                    print(f"  ✓ Relaxed filtering recovered enough lines ({len(filtered_lines)} >= {expected_line_count})")
            
            # If filtering removed everything but we had candidates, and we're using simple hough, 
            # maybe the filter was too aggressive? Fallback to raw lines for selection if needed.
            if len(filtered_lines) == 0 and len(raw_cv_lines) > 0 and settings.USE_SIMPLE_HOUGH:
                 print("  ⚠️  Filtering removed all lines. Using raw candidates for selection.")
                 filtered_lines = raw_cv_lines

            # Rank and select lines to ensure we have exactly expected_line_count
            if expected_line_count:
                if len(filtered_lines) > expected_line_count:
                    # Too many lines - select best N
                    if settings.USE_REFERENCE_BASED_MATCHING and reference_image is not None:
                        print("  🔄 Using reference-based matching to select best lines...")
                        filtered_lines = self._rank_and_select_lines_reference_based(
                            filtered_lines, expected_line_count, generated_image, product_mask, reference_image
                        )
                        print(f"  ✓ After reference-based ranking: {len(filtered_lines)} distinct lines selected")
                    else:
                        print("  🔄 Using current ranking logic to select best lines...")
                        filtered_lines = self._rank_and_select_lines(filtered_lines, expected_line_count, generated_image, product_mask)
                        print(f"  ✓ After ranking: {len(filtered_lines)} distinct lines selected")
                elif len(filtered_lines) < expected_line_count:
                    # Too few lines - still apply ranking to get best candidates, then fill from raw if needed
                    print(f"  🔄 Only {len(filtered_lines)} lines available, expected {expected_line_count}")
                    print(f"  🔄 Applying ranking to select best candidates...")
                    
                    # Rank available lines to get best ones
                    if settings.USE_REFERENCE_BASED_MATCHING and reference_image is not None:
                        ranked_lines = self._rank_and_select_lines_reference_based(
                            filtered_lines, len(filtered_lines), generated_image, product_mask, reference_image
                        )
                    else:
                        ranked_lines = self._rank_and_select_lines(
                            filtered_lines, len(filtered_lines), generated_image, product_mask
                        )
                    
                    # If we have raw lines that weren't in filtered, try to add them
                    if len(ranked_lines) < expected_line_count and len(raw_cv_lines) > len(filtered_lines):
                        print(f"  🔄 Attempting to fill remaining slots from raw candidates...")
                        # Get lines that weren't in filtered set by comparing coordinates
                        def line_key(line):
                            """Create a unique key for a line based on its coordinates"""
                            return (round(line.get('start', {}).get('x', 0), 4),
                                   round(line.get('start', {}).get('y', 0), 4),
                                   round(line.get('end', {}).get('x', 0), 4),
                                   round(line.get('end', {}).get('y', 0), 4))
                        
                        filtered_keys = {line_key(line) for line in filtered_lines}
                        remaining_candidates = [line for line in raw_cv_lines 
                                              if line_key(line) not in filtered_keys]
                        
                        # Rank remaining candidates and add best ones
                        if remaining_candidates:
                            if settings.USE_REFERENCE_BASED_MATCHING and reference_image is not None:
                                additional = self._rank_and_select_lines_reference_based(
                                    remaining_candidates, 
                                    expected_line_count - len(ranked_lines),
                                    generated_image, product_mask, reference_image
                                )
                            else:
                                additional = self._rank_and_select_lines(
                                    remaining_candidates,
                                    expected_line_count - len(ranked_lines),
                                    generated_image, product_mask
                                )
                            ranked_lines.extend(additional)
                            print(f"  ✓ Added {len(additional)} additional lines from raw candidates")
                    
                    filtered_lines = ranked_lines
                    print(f"  ✓ Final result: {len(filtered_lines)} lines (expected {expected_line_count})")
                    
                    if len(filtered_lines) < expected_line_count:
                        print(f"  ⚠️  WARNING: Only {len(filtered_lines)} lines available, cannot meet expected count of {expected_line_count}")
                else:
                    # Exactly the right count - still apply ranking to ensure best quality
                    if settings.USE_REFERENCE_BASED_MATCHING and reference_image is not None:
                        filtered_lines = self._rank_and_select_lines_reference_based(
                            filtered_lines, expected_line_count, generated_image, product_mask, reference_image
                        )
                    else:
                        filtered_lines = self._rank_and_select_lines(
                            filtered_lines, expected_line_count, generated_image, product_mask
                        )
                    print(f"  ✓ Perfect match: {len(filtered_lines)} lines (expected {expected_line_count})")
            
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
        """
        Simple Hough transform approach for extracting black lines.
        Works best for non-black products.
        
        Args:
            image: Image with black lines drawn on it
            black_threshold: Pixel value threshold for black (default: 50)
            min_line_length: Minimum line length in pixels (default: 30)
            max_line_gap: Maximum gap between line segments (default: 10)
            hough_threshold: Hough transform threshold (default: 50)
        
        Returns:
            List of lines with normalized coordinates (0-1)
        """
        img_arr = pil_to_numpy(image)
        h, w = img_arr.shape[:2]
        
        # Convert to grayscale if needed
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        # Threshold for black pixels (pixels < threshold are considered black)
        _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
        # Now: 255 = black pixels, 0 = everything else
        
        # Optional: Clean up with morphology to connect broken segments
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            binary,
            rho=1,              # Distance resolution in pixels
            theta=np.pi/180,   # Angular resolution in radians
            threshold=hough_threshold,  # Minimum votes
            minLineLength=min_line_length,  # Minimum line length
            maxLineGap=max_line_gap        # Maximum gap between segments
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

    def _extend_line_to_full_extent(
        self,
        line: dict,
        image: Image.Image,
        binary_mask: np.ndarray
    ) -> dict:
        """
        Extend a detected line segment along its direction to find the full extent of the line.
        
        Args:
            line: Line dict with normalized coordinates (0-1)
            image: PIL Image
            binary_mask: Binary mask where 255 = line pixels, 0 = background
            
        Returns:
            Extended line dict with normalized coordinates
        """
        h, w = binary_mask.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        start_x = int(line["start"]["x"] * w)
        start_y = int(line["start"]["y"] * h)
        end_x = int(line["end"]["x"] * w)
        end_y = int(line["end"]["y"] * h)
        
        # Calculate line direction vector
        dx = end_x - start_x
        dy = end_y - start_y
        line_length = np.sqrt(dx**2 + dy**2)
        
        # Handle edge case: zero-length line
        if line_length < 1:
            return line
        
        # Normalize direction vector
        dx_norm = dx / line_length
        dy_norm = dy / line_length
        
        # Extension parameters
        step_size = settings.LINE_EXTENSION_STEP_SIZE
        max_distance = min(settings.LINE_EXTENSION_MAX_DISTANCE, max(w, h) // 5)  # Limit to 20% of image dimension
        window_size = settings.LINE_EXTENSION_WINDOW_SIZE
        min_pixel_ratio = settings.LINE_EXTENSION_MIN_PIXEL_RATIO
        
        # Extend from start point backwards
        new_start_x, new_start_y = self._extend_line_direction(
            start_x, start_y, -dx_norm, -dy_norm, binary_mask, w, h,
            step_size, max_distance, window_size, min_pixel_ratio
        )
        
        # Extend from end point forwards
        new_end_x, new_end_y = self._extend_line_direction(
            end_x, end_y, dx_norm, dy_norm, binary_mask, w, h,
            step_size, max_distance, window_size, min_pixel_ratio
        )
        
        # Convert back to normalized coordinates
        return {
            "start": {"x": new_start_x / w, "y": new_start_y / h},
            "end": {"x": new_end_x / w, "y": new_end_y / h},
            "label": line.get("label", "")
        }
    
    def _extend_line_direction(
        self,
        start_x: int,
        start_y: int,
        dx_norm: float,
        dy_norm: float,
        binary_mask: np.ndarray,
        w: int,
        h: int,
        step_size: int,
        max_distance: int,
        window_size: int,
        min_pixel_ratio: float
    ) -> Tuple[int, int]:
        """
        Extend a line in a given direction until line pixels are no longer detected.
        
        Returns:
            Extended point (x, y) in pixel coordinates
        """
        current_x = float(start_x)
        current_y = float(start_y)
        distance_extended = 0
        half_window = window_size // 2
        
        while distance_extended < max_distance:
            # Calculate next point
            next_x = current_x + dx_norm * step_size
            next_y = current_y + dy_norm * step_size
            
            # Check if we've hit image boundaries
            if next_x < 0 or next_x >= w or next_y < 0 or next_y >= h:
                break
            
            # Sample pixels in a window around the next point
            x_int = int(round(next_x))
            y_int = int(round(next_y))
            
            # Check window bounds
            y_min = max(0, y_int - half_window)
            y_max = min(h, y_int + half_window + 1)
            x_min = max(0, x_int - half_window)
            x_max = min(w, x_int + half_window + 1)
            
            if y_max <= y_min or x_max <= x_min:
                break
            
            # Count line pixels in window
            window = binary_mask[y_min:y_max, x_min:x_max]
            line_pixel_count = np.sum(window > 0)
            total_pixels = window.size
            pixel_ratio = line_pixel_count / total_pixels if total_pixels > 0 else 0
            
            # If ratio is below threshold, stop extending
            if pixel_ratio < min_pixel_ratio:
                break
            
            # Continue extending
            current_x = next_x
            current_y = next_y
            distance_extended += step_size
        
        return int(round(current_x)), int(round(current_y))

    def _merge_collinear_segments(self, lines: List[dict], w: int, h: int) -> List[dict]:
        """
        Merge lines that are likely part of the same continuous line.
        Grouping criteria:
        1. Similar angle (orientation)
        2. Similar distance from origin (rho)
        3. Close proximity (endpoints close to each other)
        """
        if not lines:
            return []
            
        # Convert to working format with angle/rho
        working_lines = []
        for i, line in enumerate(lines):
            p1 = (line["start"]["x"] * w, line["start"]["y"] * h)
            p2 = (line["end"]["x"] * w, line["end"]["y"] * h)
            
            # Calculate angle and rho
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.degrees(np.arctan2(dy, dx)) % 180
            
            # Normal form: x*cos(theta) + y*sin(theta) = rho
            # Calculate rho (perpendicular distance from origin)
            # Use normal angle (angle + 90) for rho calculation
            theta_rad = np.radians(angle + 90)
            center_x = (p1[0] + p2[0]) / 2
            center_y = (p1[1] + p2[1]) / 2
            rho = center_x * np.cos(theta_rad) + center_y * np.sin(theta_rad)
            
            working_lines.append({
                "line": line,
                "angle": angle,
                "rho": rho,
                "p1": p1,
                "p2": p2,
                "merged": False
            })
        
        merged_lines = []
        
        # Sort by angle to group potentially similar lines
        working_lines.sort(key=lambda x: x["angle"])
        
        for i in range(len(working_lines)):
            if working_lines[i]["merged"]:
                continue
                
            current = working_lines[i]
            base_line = current
            
            # Look for lines to merge with current
            for j in range(i + 1, len(working_lines)):
                if working_lines[j]["merged"]:
                    continue
                    
                candidate = working_lines[j]
                
                # Check 1: Similar Angle
                angle_diff = abs(current["angle"] - candidate["angle"])
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff
                
                if angle_diff > 5.0: # 5 degrees tolerance
                    # Since sorted by angle, if diff is large, we can stop checking this group?
                    # No, because of wrap around (0 vs 179) and sorting. 
                    # But for now, simple check.
                    continue
                
                # Check 2: Similar segment bounds (Collinear check via Rho)
                # Rho diff should be small (pixels)
                if abs(current["rho"] - candidate["rho"]) > 10.0: # 10 pixels tolerance
                    continue
                
                # Check 3: Proximity (Gap check)
                # Project points onto the line defined by 'current' and check 1D distance?
                # Or simple endpoint distance
                # We simply check if the segments overlap or are close
                
                # Calculate extent along the line
                # Rotate points by -angle to align with X axis
                rad = np.radians(-current["angle"])
                c, s = np.cos(rad), np.sin(rad)
                
                def rotate(x, y):
                    return x * c - y * s, x * s + y * c
                
                cp1_proj = rotate(*current["p1"])[0]
                cp2_proj = rotate(*current["p2"])[0]
                cand_p1_proj = rotate(*candidate["p1"])[0]
                cand_p2_proj = rotate(*candidate["p2"])[0]
                
                min_c, max_c = min(cp1_proj, cp2_proj), max(cp1_proj, cp2_proj)
                min_cand, max_cand = min(cand_p1_proj, cand_p2_proj), max(cand_p1_proj, cand_p2_proj)
                
                # Check overlap or gap
                # Overlap: max(start1, start2) < min(end1, end2)
                # Gap: max(start1, start2) - min(end1, end2)
                
                dist_between = max(min_c, min_cand) - min(max_c, max_cand)
                
                if dist_between < 50: # 50 pixels gap tolerance (matches Hough gap)
                    # MERGE
                    working_lines[j]["merged"] = True
                    
                    # Update 'current' to include this line essentially
                    # We need to find the extreme points of the merged line
                    all_points = [current["p1"], current["p2"], candidate["p1"], candidate["p2"]]
                    
                    # Project all to axis to find extremes
                    projs = [rotate(*p)[0] for p in all_points]
                    min_idx = np.argmin(projs)
                    max_idx = np.argmax(projs)
                    
                    current["p1"] = all_points[min_idx]
                    current["p2"] = all_points[max_idx]
                    
                    # Recalculate rho/angle? Maybe average? keep original for stability
            
            # Create final parsed line
            merged_lines.append({
                "start": {"x": current["p1"][0] / w, "y": current["p1"][1] / h},
                "end": {"x": current["p2"][0] / w, "y": current["p2"][1] / h},
                "label": ""
            })
            
        return merged_lines



    def extract_lines_from_generated_image_simple(
        self,
        generated_image: Image.Image,
        expected_line_count: Optional[int] = None,
        black_threshold: int = 50,
        min_line_length: int = 30,
        max_line_gap: int = 10,
        hough_threshold: int = 50
    ) -> List[dict]:
        """
        Simple extraction using Hough transform only.
        No product mask, no filtering, no complex logic.
        Best for non-black products with clear black lines.
        
        Args:
            generated_image: Image with black lines drawn on it
            expected_line_count: Optional expected number of lines (for filtering)
            black_threshold: Pixel value threshold for black (default: 50)
            min_line_length: Minimum line length in pixels (default: 30)
            max_line_gap: Maximum gap between line segments (default: 10)
            hough_threshold: Hough transform threshold (default: 50)
        
        Returns:
            List of lines with normalized coordinates (0-1)
        """
        print("  🔄 Using simple Hough transform extraction...")
        
        # Extract lines using simple Hough
        lines = self.extract_lines_simple_hough(
            generated_image,
            black_threshold=black_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            hough_threshold=hough_threshold
        )
        
        print(f"  ✓ Simple Hough found {len(lines)} lines")
        
        # Optional: Filter by expected count if provided
        if expected_line_count and len(lines) > expected_line_count:
            print(f"  ⚠️  Found {len(lines)} lines, expected {expected_line_count}")
            # Simple ranking: sort by line length
            h, w = generated_image.size[1], generated_image.size[0]
            scored_lines = []
            for line in lines:
                start_x = line["start"]["x"] * w
                start_y = line["start"]["y"] * h
                end_x = line["end"]["x"] * w
                end_y = line["end"]["y"] * h
                length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                scored_lines.append((length, line))
            
            # Sort by length (descending)
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            candidate_lines = [line for _, line in scored_lines]
            
            # Select lines that are DISTINCT (not similar to already selected)
            selected_lines = []
            for line in candidate_lines:
                # Stop if we have enough lines
                if expected_line_count and len(selected_lines) >= expected_line_count:
                    break
                
                # Check similarity with already selected lines
                is_distinct = True
                for selected in selected_lines:
                    if self._are_lines_similar(line, selected, generated_image):
                        is_distinct = False
                        break
                
                if is_distinct:
                    selected_lines.append(line)
            
            lines = selected_lines
            print(f"  ✓ Selected top {len(lines)} distinct longest lines")
        
        return lines
    
    # [DEPRECATED] _extract_via_diff, _extract_via_cv, _merge_lines_cv_only removed.

    
    def _refine_with_edges(
        self,
        line: dict,
        original_image: Image.Image
    ) -> dict:
        """Refine line coordinates by snapping to edges"""
        img_arr = pil_to_numpy(original_image)
        edges = detect_edges(img_arr)
        h, w = img_arr.shape[:2]
        
        # Snap start and end points
        start_pixel = (
            int(line["start"]["x"] * w),
            int(line["start"]["y"] * h)
        )
        end_pixel = (
            int(line["end"]["x"] * w),
            int(line["end"]["y"] * h)
        )
        
        refined_start = snap_point_to_edge(start_pixel, edges)
        refined_end = snap_point_to_edge(end_pixel, edges)
        
        return {
            "start": {"x": refined_start[0] / w, "y": refined_start[1] / h},
            "end": {"x": refined_end[0] / w, "y": refined_end[1] / h},
            "label": line.get("label", "")
        }
    

    def _extract_via_cv(self, image: Image.Image, product_mask: Optional[np.ndarray] = None) -> List[dict]:
        """
        Extract lines using color-based detection (product-aware).
        Falls back to edge-based detection if color-based fails.
        """
        img_arr = pil_to_numpy(image)
        
        # Try color-based detection first (more targeted for black lines)
        try:
            lines = detect_lines_by_color(img_arr, product_mask)
        except Exception as e:
            print(f"  ⚠️  Color-based detection failed: {e}, falling back to edge-based")
            # Fallback to edge-based detection
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
        """
        Detect product mask (binary: 1=product, 0=background)
        Assumes white background - product is the main non-white object
        """
        img_arr = pil_to_numpy(original_image)
        h, w = img_arr.shape[:2]
        
        # Convert to grayscale if needed
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        # Threshold to separate product from white background
        # White background should be close to 255, product should be darker
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: return empty mask (all background)
            return np.zeros((h, w), dtype=np.uint8)
        
        # Find largest contour (assuming product is the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from largest contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [largest_contour], 255)
        
        # Optional: dilate slightly to include product edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _filter_lines_outside_product(
        self,
        lines: List[dict],
        product_mask: np.ndarray,
        original_image: Image.Image
    ) -> List[dict]:
        """
        Filter lines that are outside product mask/contour
        Returns only lines where majority of points are outside the product
        """
        h, w = product_mask.shape[:2]
        filtered_lines = []
        
        for line in lines:
            # Get line points in pixel coordinates
            start_x = int(line["start"]["x"] * w)
            start_y = int(line["start"]["y"] * h)
            end_x = int(line["end"]["x"] * w)
            end_y = int(line["end"]["y"] * h)
            
            # Clamp to image bounds
            start_x = max(0, min(w - 1, start_x))
            start_y = max(0, min(h - 1, start_y))
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Check if start and end points are outside product mask
            start_inside = product_mask[start_y, start_x] > 0 if (0 <= start_y < h and 0 <= start_x < w) else False
            end_inside = product_mask[end_y, end_x] > 0 if (0 <= end_y < h and 0 <= end_x < w) else False
            
            # Check midpoint
            mid_x = int((start_x + end_x) / 2)
            mid_y = int((start_y + end_y) / 2)
            mid_x = max(0, min(w - 1, mid_x))
            mid_y = max(0, min(h - 1, mid_y))
            mid_inside = product_mask[mid_y, mid_x] > 0 if (0 <= mid_y < h and 0 <= mid_x < w) else False
            
            # Sample a few points along the line
            num_samples = 5
            inside_count = 0
            total_samples = 0
            
            for i in range(num_samples + 1):
                t = i / num_samples
                x = int(start_x * (1 - t) + end_x * t)
                y = int(start_y * (1 - t) + end_y * t)
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                if 0 <= y < h and 0 <= x < w:
                    total_samples += 1
                    if product_mask[y, x] > 0:
                        inside_count += 1
            
            # Line is considered outside if majority of points are outside
            # Allow some overlap (e.g., up to 30% of line can be inside)
            outside_ratio = 1.0 - (inside_count / total_samples) if total_samples > 0 else 1.0
            
            if outside_ratio >= 0.7:  # At least 70% of line is outside product
                filtered_lines.append(line)
        
        return filtered_lines
    
    def _filter_measurement_lines(
        self,
        lines: List[dict],
        product_mask: np.ndarray,
        generated_image: Image.Image
    ) -> List[dict]:
        """
        Filter to keep only measurement lines (both inside and outside product)
        Measurement lines are:
        - Black lines drawn by Gemini (high contrast)
        - Positioned with clear separation from product edges (for outside lines)
        - Longer and more prominent than artifacts
        """
        h, w = product_mask.shape[:2]
        img_arr = pil_to_numpy(generated_image)
        
        # Convert to grayscale for contrast analysis
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        filtered_lines = []
        
        # Calculate distance transform to find distance from product edges
        mask_for_dist = product_mask.astype(np.uint8)
        if len(mask_for_dist.shape) > 2:
            mask_for_dist = cv2.cvtColor(mask_for_dist, cv2.COLOR_RGB2GRAY)
        
        inverted_mask = cv2.bitwise_not(mask_for_dist)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        
        # Calculate minimum line length threshold
        min_length = min(w, h) * settings.MEASUREMENT_LINE_MIN_LENGTH_RATIO
        
        for line in lines:
            # Get line points in pixel coordinates
            start_x = int(line["start"]["x"] * w)
            start_y = int(line["start"]["y"] * h)
            end_x = int(line["end"]["x"] * w)
            end_y = int(line["end"]["y"] * h)
            
            # Clamp to image bounds
            start_x = max(0, min(w - 1, start_x))
            start_y = max(0, min(h - 1, start_y))
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Calculate line length
            line_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # Filter 1: Skip very short lines (likely artifacts)
            if line_length < min_length:
                continue
            
            # Sample points along line to analyze
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
                    # Check if inside product
                    if product_mask[y, x] > 0:
                        inside_count += 1
                    
                    # Get distance to product edge
                    dist_to_edge = dist_transform[y, x]
                    min_distance_to_edge = min(min_distance_to_edge, dist_to_edge)
                    
                    # Check contrast (measurement lines should be very dark/black)
                    pixel_value = gray[y, x]
                    contrast_values.append(pixel_value)
            
            if len(contrast_values) == 0:
                continue
            
            inside_ratio = inside_count / (num_samples + 1)
            outside_ratio = 1.0 - inside_ratio
            avg_contrast = np.mean(contrast_values)
            max_contrast = np.max(contrast_values)
            
            # Filter 2: High contrast check (must be very dark/black)
            is_high_contrast = (avg_contrast < settings.MEASUREMENT_LINE_CONTRAST_THRESHOLD and 
                              max_contrast < settings.MEASUREMENT_LINE_CONTRAST_MAX)
            
            if not is_high_contrast:
                continue
            
            # Filter 3: Position filter
            # - Outside lines: >=70% outside AND minimum distance from product edge
            # - Inside lines: >=70% inside (no distance requirement, as they may be between legs)
            # - Exclude: Lines that are 30-70% inside/outside (likely product edges)
            
            is_clear_position = (inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD or 
                               outside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD)
            
            if not is_clear_position:
                continue
            
            # For outside lines, require minimum separation from product edge
            if outside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD:
                if min_distance_to_edge < settings.MEASUREMENT_LINE_EDGE_SEPARATION:
                    continue  # Too close to product edge, likely a product edge
            
            # Keep line if it meets all criteria
            filtered_lines.append(line)
        
        return filtered_lines
    
    def _filter_measurement_lines_relaxed(
        self,
        lines: List[dict],
        product_mask: np.ndarray,
        generated_image: Image.Image
    ) -> List[dict]:
        """
        Relaxed version of _filter_measurement_lines with looser criteria.
        Used when strict filtering removes too many lines.
        """
        h, w = product_mask.shape[:2]
        img_arr = pil_to_numpy(generated_image)
        
        # Convert to grayscale for contrast analysis
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr
        
        filtered_lines = []
        
        # Calculate distance transform to find distance from product edges
        mask_for_dist = product_mask.astype(np.uint8)
        if len(mask_for_dist.shape) > 2:
            mask_for_dist = cv2.cvtColor(mask_for_dist, cv2.COLOR_RGB2GRAY)
        
        inverted_mask = cv2.bitwise_not(mask_for_dist)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        
        # RELAXED: Lower minimum line length (50% of strict threshold)
        min_length = min(w, h) * settings.MEASUREMENT_LINE_MIN_LENGTH_RATIO * 0.5
        
        # RELAXED: Higher contrast threshold (allow slightly lighter lines)
        relaxed_contrast_threshold = settings.MEASUREMENT_LINE_CONTRAST_THRESHOLD + 30  # Allow up to 100 instead of 70
        relaxed_contrast_max = settings.MEASUREMENT_LINE_CONTRAST_MAX + 30  # Allow up to 130 instead of 100
        
        # RELAXED: Lower position threshold (50% instead of 60%)
        relaxed_position_threshold = 0.5
        
        for line in lines:
            # Get line points in pixel coordinates
            start_x = int(line["start"]["x"] * w)
            start_y = int(line["start"]["y"] * h)
            end_x = int(line["end"]["x"] * w)
            end_y = int(line["end"]["y"] * h)
            
            # Clamp to image bounds
            start_x = max(0, min(w - 1, start_x))
            start_y = max(0, min(h - 1, start_y))
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Calculate line length
            line_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # RELAXED Filter 1: Skip very short lines (but with lower threshold)
            if line_length < min_length:
                continue
            
            # Sample points along line to analyze
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
                    # Check if inside product
                    if product_mask[y, x] > 0:
                        inside_count += 1
                    
                    # Get distance to product edge
                    dist_to_edge = dist_transform[y, x]
                    min_distance_to_edge = min(min_distance_to_edge, dist_to_edge)
                    
                    # Check contrast
                    pixel_value = gray[y, x]
                    contrast_values.append(pixel_value)
            
            if len(contrast_values) == 0:
                continue
            
            inside_ratio = inside_count / (num_samples + 1)
            outside_ratio = 1.0 - inside_ratio
            avg_contrast = np.mean(contrast_values)
            max_contrast = np.max(contrast_values)
            
            # RELAXED Filter 2: More lenient contrast check
            is_high_contrast = (avg_contrast < relaxed_contrast_threshold and 
                              max_contrast < relaxed_contrast_max)
            
            if not is_high_contrast:
                continue
            
            # RELAXED Filter 3: Lower position threshold
            is_clear_position = (inside_ratio >= relaxed_position_threshold or 
                               outside_ratio >= relaxed_position_threshold)
            
            if not is_clear_position:
                continue
            
            # RELAXED: For outside lines, no minimum separation requirement (or very low)
            # (We skip the edge separation check in relaxed mode)
            
            # Keep line if it meets relaxed criteria
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
                if abs(line1_dx) < 1e-6:  # Vertical line (line1)
                    # Both lines are vertical (since they're parallel)
                    perp_distance = abs(line1_mid_x - line2_mid_x) / w
                elif abs(line1_dy) < 1e-6:  # Horizontal line (line1)
                    # Both lines are horizontal (since they're parallel)
                    perp_distance = abs(line1_mid_y - line2_mid_y) / h
                elif abs(line2_dx) < 1e-6:  # Vertical line (line2) - shouldn't happen if parallel, but handle it
                    perp_distance = abs(line1_mid_x - line2_mid_x) / w
                elif abs(line2_dy) < 1e-6:  # Horizontal line (line2) - shouldn't happen if parallel, but handle it
                    perp_distance = abs(line1_mid_y - line2_mid_y) / h
                else:
                    # General line: y = mx + b (both lines are neither vertical nor horizontal)
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
            
            # Combined score: normalize length and distance, then combine
            # Normalize by image diagonal for length
            max_length = np.sqrt(w**2 + h**2)
            normalized_length = line_length / max_length if max_length > 0 else 0
            
            # Normalize distance (assuming max distance is half image size)
            max_distance = min(w, h) / 2
            normalized_distance = avg_distance / max_distance if max_distance > 0 else 0
            
            # Combined score (weighted: 60% length, 40% distance)
            score = 0.6 * normalized_length + 0.4 * normalized_distance
            
            scored_lines.append({
                "line": line,
                "score": score,
                "length": line_length,
                "distance": avg_distance
            })
        
        # Sort by score (descending)
        scored_lines.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top N DISTINCT lines (avoid duplicates)
        selected = []
        for item in scored_lines:
            if len(selected) >= expected_count:
                break
            
            candidate_line = item["line"]
            is_distinct = True
            
            # Check if candidate is similar to any already selected line (in this batch)
            for selected_line in selected:
                if self._are_lines_similar(candidate_line, selected_line, original_image):
                    is_distinct = False
                    break
            
            # Check if candidate is similar to any ALREADY EXISTING line (from previous steps)
            if is_distinct and existing_lines:
                for existing_line in existing_lines:
                    if self._are_lines_similar(candidate_line, existing_line, original_image):
                        is_distinct = False
                        break
            
            # Only add if distinct from all previously selected lines
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
        """
        Rank and select lines by matching to reference image pattern.
        This is the new reference-based matching approach.
        """
        if len(lines) <= expected_count:
            return lines
        
        # Step 1: Extract lines from reference image
        print("    📐 Extracting lines from reference image...")
        ref_product_mask = self._detect_product_mask(reference_image)
        ref_cv_lines = self._extract_via_cv(reference_image, ref_product_mask)
        ref_filtered_lines = self._filter_measurement_lines(ref_cv_lines, ref_product_mask, reference_image)
        
        if not ref_filtered_lines:
            print("    ⚠️  No lines found in reference image, falling back to current logic")
            return self._rank_and_select_lines(lines, expected_count, generated_image, product_mask)
        
        print(f"    ✓ Found {len(ref_filtered_lines)} reference lines")
        
        # Step 2: Classify reference lines (inside/outside)
        ref_lines_classified = []
        for ref_line in ref_filtered_lines:
            inside_ratio = self._calculate_inside_ratio(ref_line, ref_product_mask, reference_image)
            is_inside = inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD
            ref_lines_classified.append({
                "line": ref_line,
                "is_inside": is_inside,
                "inside_ratio": inside_ratio
            })
        
        # Count inside vs outside in reference
        ref_inside_count = sum(1 for rl in ref_lines_classified if rl["is_inside"])
        ref_outside_count = len(ref_lines_classified) - ref_inside_count
        print(f"    📊 Reference pattern: {ref_inside_count} inside, {ref_outside_count} outside")
        
        # Step 3: For each reference line, find best matching generated line
        # If reference has more lines than expected, prioritize by length (longer = more important)
        if len(ref_lines_classified) > expected_count:
            print(f"    ⚠️  Reference has {len(ref_lines_classified)} lines, but only {expected_count} expected")
            print(f"    🔄 Prioritizing longest reference lines for matching...")
            # Sort reference lines by length (longer = more important)
            ref_lines_with_length = []
            ref_h, ref_w = ref_product_mask.shape[:2]
            for ref_item in ref_lines_classified:
                ref_line = ref_item["line"]
                start_x = ref_line["start"]["x"] * reference_image.size[0]
                start_y = ref_line["start"]["y"] * reference_image.size[1]
                end_x = ref_line["end"]["x"] * reference_image.size[0]
                end_y = ref_line["end"]["y"] * reference_image.size[1]
                length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                ref_lines_with_length.append((length, ref_item))
            # Sort by length descending
            ref_lines_with_length.sort(key=lambda x: x[0], reverse=True)
            # Take only the top expected_count reference lines
            ref_lines_classified = [item for _, item in ref_lines_with_length[:expected_count]]
            print(f"    ✓ Using top {len(ref_lines_classified)} longest reference lines")
        
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
            
            # Find best matching generated line
            best_match = None
            best_score = -1
            best_idx = -1
            
            for gen_idx, gen_line in enumerate(lines):
                if gen_idx in used_generated_indices:
                    continue
                
                # Filter 1: Must match inside/outside
                gen_inside_ratio = self._calculate_inside_ratio(gen_line, product_mask, generated_image)
                gen_is_inside = gen_inside_ratio >= settings.MEASUREMENT_LINE_POSITION_THRESHOLD
                
                if gen_is_inside != ref_is_inside:
                    continue  # Skip if inside/outside doesn't match
                
                # Calculate match score
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
            
            # Add best match if found and not duplicate of already selected lines
            if best_match is not None and best_score > 0:
                # Check if this line is too similar to any already selected line
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
                    # Try to find next best match that's not a duplicate
                    # Remove this candidate and continue searching
                    pass
            else:
                print(f"    ⚠️  No match found for reference line (inside={ref_is_inside})")
        
        # If we didn't get enough matches, fill remaining slots with current logic
        if len(selected) < expected_count:
            remaining_needed = expected_count - len(selected)
            print(f"    ⚠️  Only found {len(selected)} matches, filling remaining {remaining_needed} slots with current logic")
            remaining_lines = [line for idx, line in enumerate(lines) if idx not in used_generated_indices]
            if remaining_lines:
                # Try to get the needed count, but if we can't find enough distinct lines,
                # we'll take what we can get (but log a warning)
                remaining_selected = self._rank_and_select_lines(
                    remaining_lines, remaining_needed, generated_image, product_mask,
                    existing_lines=selected
                )
                selected.extend(remaining_selected)
                
                # If we still don't have enough, try to be less strict about duplicates
                if len(selected) < expected_count and len(remaining_lines) > len(remaining_selected):
                    print(f"    ⚠️  Only got {len(remaining_selected)} additional lines, need {remaining_needed - len(remaining_selected)} more")
                    print(f"    🔄 Attempting to fill remaining slots with less strict duplicate checking...")
                    
                    # Get lines that weren't selected (might be similar but still valid)
                    remaining_candidates = [line for line in remaining_lines if line not in remaining_selected]
                    
                    # Try to add more lines even if they're somewhat similar
                    for candidate in remaining_candidates:
                        if len(selected) >= expected_count:
                            break
                        
                        # Check if it's too similar to already selected (but be more lenient)
                        is_too_similar = False
                        for existing in selected:
                            if self._are_lines_similar(candidate, existing, generated_image):
                                is_too_similar = True
                                break
                        
                        if not is_too_similar:
                            selected.append(candidate)
                            print(f"    ✓ Added additional line to reach expected count")
                    
                    if len(selected) < expected_count:
                        print(f"    ⚠️  WARNING: Could only get {len(selected)} lines, expected {expected_count}")
        
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

