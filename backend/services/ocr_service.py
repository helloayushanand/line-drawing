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
        return extracted_lines


