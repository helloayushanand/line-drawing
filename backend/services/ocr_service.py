import easyocr
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from utils.image_utils import pil_to_numpy


class OCRService:
    def __init__(self):
        """Initialize EasyOCR reader with English language support"""
        print("🔄 Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
        print("✓ EasyOCR reader initialized")
    
    def _preprocess_for_ocr(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to numpy array for OCR.
        Preprocessing disabled as per user request.
        """
        # Convert PIL to numpy
        img_arr = pil_to_numpy(image)
        
        return img_arr
    
    def extract_labels(
        self, 
        image: Image.Image,
        expected_labels: Optional[List[str]] = None
    ) -> List[dict]:
        """
        Extract text labels and their positions from image.
        
        Args:
            image: PIL Image to extract labels from
            expected_labels: Optional list of expected label patterns (e.g., ["Line 1", "Line 2"])
        
        Returns:
            List of dicts with format:
            [
                {
                    "text": "Line 1",
                    "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 4 corner points
                    "center": (x, y),  # Normalized 0-1
                    "confidence": 0.95
                },
                ...
            ]
        """
        print("\n🔍 Extracting labels from image using OCR...")
        
        # Preprocess image for better OCR accuracy
        print("  🔄 Preprocessing image for OCR...")
        preprocessed = self._preprocess_for_ocr(image)
        
        # Get original dimensions for normalization
        h, w = preprocessed.shape[:2]
        
        # Run OCR with adjusted parameters
        # contrast_ths: lower = more sensitive to low contrast text
        # adjust_contrast: enhance contrast before detection
        results = self.reader.readtext(
            preprocessed,
            contrast_ths=0.1,      # Lower threshold for low-contrast text (default: 0.5)
            adjust_contrast=0.8,   # Adjust contrast (default: 0.5)
            text_threshold=0.5,    # Text detection threshold (default: 0.7)
            low_text=0.3          # Low text threshold (default: 0.4)
        )
        
        print(f"  ✓ OCR detected {len(results)} text regions")
        
        # Process results
        labels = []
        for bbox, text, confidence in results:
            # Lower confidence threshold to catch more detections
            if confidence < 0.3:  # Lowered from 0.5
                print(f"  ⚠️  Skipping low-confidence text: '{text}' (confidence: {confidence:.2f})")
                continue
            
            # Optional: Filter by expected label patterns
            if expected_labels:
                # Check if text matches any expected label (case-insensitive)
                text_lower = text.lower().strip()
                matches = any(expected.lower() in text_lower or text_lower in expected.lower() 
                             for expected in expected_labels)
                if not matches:
                    print(f"  ⚠️  Skipping unexpected text: '{text}'")
                    continue
            
            # Calculate center point from bounding box
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            bbox_array = np.array(bbox)
            center_x = np.mean(bbox_array[:, 0]) / w  # Normalize to 0-1
            center_y = np.mean(bbox_array[:, 1]) / h  # Normalize to 0-1
            
            labels.append({
                "text": text.strip(),
                "bbox": bbox,
                "center": (center_x, center_y),
                "confidence": confidence
            })
            
            print(f"  ✓ Extracted label: '{text}' at ({center_x:.3f}, {center_y:.3f}), confidence: {confidence:.2f}")
        
        print(f"✓ Extracted {len(labels)} valid labels")
        return labels
    
    def match_lines_to_labels(
        self,
        extracted_lines: List[dict],
        ocr_labels: List[dict],
        reference_lines: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Match each extracted line to nearest OCR label by proximity.
        Uses fuzzy matching to handle partial OCR detections.
        
        Args:
            extracted_lines: Lines extracted from generated image (CV-based)
                Format: [{"start": {"x": ..., "y": ...}, "end": {...}, "label": ""}, ...]
            ocr_labels: Labels extracted via OCR
                Format: [{"text": "Line 1", "center": (x, y), ...}, ...]
            reference_lines: Original reference lines with labels (for fuzzy matching)
        
        Returns:
            Lines with matched labels assigned
        """
        print("\n🔗 Matching extracted lines to OCR labels...")
        
        if not ocr_labels:
            print("  ⚠️  No OCR labels found, returning lines without labels")
            return extracted_lines
        
        # Build reference label mapping for fuzzy matching
        ref_label_map = {}
        if reference_lines:
            for ref_line in reference_lines:
                label = ref_line.get('label', '')
                if label:
                    ref_label_map[label.lower().strip()] = label
        
        print(f"  📋 Reference labels: {list(ref_label_map.values())}")
        print(f"  🔍 OCR detected labels: {[l['text'] for l in ocr_labels]}")
        
        # Normalize OCR labels to reference labels
        normalized_ocr_labels = []
        for ocr_label in ocr_labels:
            ocr_text = ocr_label['text'].strip()
            matched_ref_label = None
            
            # Try exact match first
            if ocr_text.lower() in ref_label_map:
                matched_ref_label = ref_label_map[ocr_text.lower()]
            else:
                # Try fuzzy match (partial match)
                matches = []
                for ref_key, ref_value in ref_label_map.items():
                    if (ocr_text.lower() in ref_key or ref_key in ocr_text.lower()):
                        matches.append(ref_value)
                
                # Only accept match if it's unique
                if len(matches) == 1:
                    matched_ref_label = matches[0]
                    print(f"  🔄 Fuzzy matched '{ocr_text}' → '{matched_ref_label}'")
            
            if matched_ref_label:
                normalized_ocr_labels.append({
                    **ocr_label,
                    'normalized_text': matched_ref_label,
                    'is_valid_ref': True
                })
            else:
                # Still include it as a potential anchor, but mark as not a valid ref
                normalized_ocr_labels.append({
                    **ocr_label,
                    'normalized_text': ocr_text,
                    'is_valid_ref': False
                })
        
        # Greedy Matching: Calculate all pairwise distances
        # Pairs: (line_index, label_index, distance)
        possible_matches = []
        
        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            """Calculate minimum distance from point (px,py) to line segment ((x1,y1), (x2,y2))"""
            # Vector from start to end of segment
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)

            # Project point onto line (parameter t)
            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)

            # Clamp t to segment [0, 1]
            t = max(0, min(1, t))

            # Closest point on segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

            return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        for i, line in enumerate(extracted_lines):
            # Get line coordinates
            x1, y1 = line["start"]["x"], line["start"]["y"]
            x2, y2 = line["end"]["x"], line["end"]["y"]
            
            for j, label in enumerate(normalized_ocr_labels):
                if not label['is_valid_ref']:
                    continue
                    
                label_x, label_y = label["center"]
                
                # Calculate proper point-to-segment distance
                distance = point_to_segment_distance(label_x, label_y, x1, y1, x2, y2)
                
                possible_matches.append({
                    'line_idx': i,
                    'label_idx': j,
                    'distance': distance,
                    'label_text': label['normalized_text']
                })
        
        # Sort by distance (closest matches first)
        possible_matches.sort(key=lambda x: x['distance'])
        
        # Assign unique matches
        matched_line_indices = set()
        matched_labels = set()
        
        # Initialize all lines with empty label
        for line in extracted_lines:
            line['label'] = ''
            
        print("\n  🔗 Assigning labels (greedy unique matching):")
        for match in possible_matches:
            line_idx = match['line_idx']
            label_text = match['label_text']
            
            # If line already matched or label already used, skip
            if line_idx in matched_line_indices or label_text in matched_labels:
                continue
                
            # Assign match
            extracted_lines[line_idx]['label'] = label_text
            matched_line_indices.add(line_idx)
            matched_labels.add(label_text)
            print(f"    ✓ Line {line_idx+1} matched to '{label_text}' (dist: {match['distance']:.3f})")
            
        print(f"✓ Matched {len(matched_line_indices)} lines to unique labels")
        
        # Assign unmatched reference labels to unmatched lines
        if reference_lines:
            unmatched_line_indices = [i for i, line in enumerate(extracted_lines) 
                                    if not line.get('label', '')]
            unmatched_ref_labels = [label for label in ref_label_map.values() 
                                   if label not in matched_labels]
            
            if unmatched_line_indices and unmatched_ref_labels:
                print(f"\n  🔄 Assigning unmatched labels to unmatched lines...")
                print(f"    Unmatched lines: {len(unmatched_line_indices)}")
                print(f"    Unmatched labels: {unmatched_ref_labels}")
                
                # Build reference line positions map (label -> line midpoint)
                ref_line_positions = {}
                for ref_line in reference_lines:
                    label = ref_line.get('label', '')
                    if label and label in unmatched_ref_labels:
                        # Calculate midpoint of reference line
                        start = ref_line.get('start', {})
                        end = ref_line.get('end', {})
                        if start and end:
                            mid_x = (start.get('x', 0) + end.get('x', 0)) / 2
                            mid_y = (start.get('y', 0) + end.get('y', 0)) / 2
                            ref_line_positions[label] = (mid_x, mid_y)
                
                # Calculate distances between unmatched lines and unmatched labels
                unmatched_matches = []
                for line_idx in unmatched_line_indices:
                    line = extracted_lines[line_idx]
                    # Calculate line midpoint
                    line_mid_x = (line["start"]["x"] + line["end"]["x"]) / 2
                    line_mid_y = (line["start"]["y"] + line["end"]["y"]) / 2
                    
                    for label in unmatched_ref_labels:
                        if label in ref_line_positions:
                            ref_mid_x, ref_mid_y = ref_line_positions[label]
                            # Calculate Euclidean distance
                            distance = np.sqrt((line_mid_x - ref_mid_x)**2 + (line_mid_y - ref_mid_y)**2)
                            unmatched_matches.append({
                                'line_idx': line_idx,
                                'label': label,
                                'distance': distance
                            })
                        else:
                            # If no reference line position, use a large distance
                            # This will be matched last
                            unmatched_matches.append({
                                'line_idx': line_idx,
                                'label': label,
                                'distance': 999.0
                            })
                
                # Sort by distance (closest first)
                unmatched_matches.sort(key=lambda x: x['distance'])
                
                # Assign unique matches (greedy)
                used_line_indices = set()
                used_labels = set()
                
                for match in unmatched_matches:
                    line_idx = match['line_idx']
                    label = match['label']
                    
                    if line_idx in used_line_indices or label in used_labels:
                        continue
                    
                    extracted_lines[line_idx]['label'] = label
                    used_line_indices.add(line_idx)
                    used_labels.add(label)
                    print(f"    ✓ Assigned unmatched label '{label}' to line {line_idx+1} (dist: {match['distance']:.3f})")
                
                # If there are still unmatched lines and labels, assign remaining labels in order
                remaining_unmatched_lines = [i for i in unmatched_line_indices if i not in used_line_indices]
                remaining_unmatched_labels = [l for l in unmatched_ref_labels if l not in used_labels]
                
                if remaining_unmatched_lines and remaining_unmatched_labels:
                    print(f"    🔄 Assigning remaining {len(remaining_unmatched_labels)} labels to {len(remaining_unmatched_lines)} lines...")
                    for i, line_idx in enumerate(remaining_unmatched_lines):
                        if i < len(remaining_unmatched_labels):
                            label = remaining_unmatched_labels[i]
                            extracted_lines[line_idx]['label'] = label
                            print(f"    ✓ Assigned remaining label '{label}' to line {line_idx+1}")
        
        # GUARANTEE: Ensure count matches reference_lines and all lines have labels
        if reference_lines:
            expected_count = len(reference_lines)
            current_count = len(extracted_lines)
            ref_labels = [line.get('label', '') for line in reference_lines if line.get('label', '')]
            
            # Note: We can't create new lines here - that's a detection problem
            # But we CAN ensure all existing lines have labels
            if current_count < expected_count:
                print(f"  ⚠️  GUARANTEE: Only {current_count} lines detected, expected {expected_count}")
                print(f"  ⚠️  This indicates a detection issue - cannot create missing lines")
            
            # GUARANTEE: Ensure ALL existing lines have labels from reference_lines
            lines_without_labels = [i for i, line in enumerate(extracted_lines) if not line.get('label', '')]
            if lines_without_labels and ref_labels:
                print(f"  🔄 GUARANTEE: Assigning labels to {len(lines_without_labels)} lines without labels...")
                
                # Get labels that haven't been used yet
                used_labels = {line.get('label', '') for line in extracted_lines if line.get('label', '')}
                available_labels = [label for label in ref_labels if label not in used_labels]
                
                # Assign available labels to lines without labels
                for i, line_idx in enumerate(lines_without_labels):
                    if i < len(available_labels):
                        extracted_lines[line_idx]['label'] = available_labels[i]
                        print(f"  ✓ Assigned label '{available_labels[i]}' to line {line_idx+1}")
                    elif ref_labels:
                        # If we've used all available unique labels, reuse them in order
                        label = ref_labels[i % len(ref_labels)]
                        extracted_lines[line_idx]['label'] = label
                        print(f"  ✓ Assigned label '{label}' to line {line_idx+1} (reused)")
            
            # Final validation
            final_count = len(extracted_lines)
            lines_with_labels_final = [line for line in extracted_lines if line.get('label', '')]
            
            if len(lines_with_labels_final) == final_count:
                print(f"  ✓ GUARANTEE: All {final_count} lines have labels")
            else:
                print(f"  ⚠️  GUARANTEE: {final_count - len(lines_with_labels_final)} lines still without labels")
        
        return extracted_lines


