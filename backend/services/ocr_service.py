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
        """Convert PIL image to numpy array for OCR"""
        img_arr = pil_to_numpy(image)
        
        try:
            from datetime import datetime
            import os
            debug_dir = os.path.join(os.path.dirname(__file__), "..", "generated_images")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = os.path.join(debug_dir, f"debug_ocr_input_{timestamp}.png")
            Image.fromarray(img_arr).save(debug_path)
            print(f"  💾 Saved debug OCR input to: {debug_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save debug OCR image: {e}")
        
        return img_arr
    
    def extract_labels(
        self, 
        image: Image.Image,
        expected_labels: Optional[List[str]] = None
    ) -> List[dict]:
        """Extract text labels and their positions from image"""
        print("\n🔍 Extracting labels from image using OCR...")
        print("  🔄 Preprocessing image for OCR...")
        preprocessed = self._preprocess_for_ocr(image)
        h, w = preprocessed.shape[:2]
        
        results = self.reader.readtext(
            preprocessed,
            contrast_ths=0.1,
            adjust_contrast=0.8,
            text_threshold=0.5,
            low_text=0.3
        )
        
        print(f"  ✓ OCR detected {len(results)} text regions")
        
        labels = []
        for bbox, text, confidence in results:
            if confidence < 0.3:
                print(f"  ⚠️  Skipping low-confidence text: '{text}' (confidence: {confidence:.2f})")
                continue
            
            if expected_labels:
                text_lower = text.lower().strip()
                matches = any(expected.lower() in text_lower or text_lower in expected.lower() 
                             for expected in expected_labels)
                if not matches:
                    print(f"  ⚠️  Skipping unexpected text: '{text}'")
                    continue
            
            bbox_array = np.array(bbox)
            center_x = np.mean(bbox_array[:, 0]) / w
            center_y = np.mean(bbox_array[:, 1]) / h
            
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
        """Match extracted lines to OCR labels by proximity"""
        print("\n🔗 Matching extracted lines to OCR labels...")
        
        if not ocr_labels:
            print("  ⚠️  No OCR labels found, returning lines without labels")
            return extracted_lines
        
        ref_label_map = {}
        if reference_lines:
            for ref_line in reference_lines:
                label = ref_line.get('label', '')
                if label:
                    ref_label_map[label.lower().strip()] = label
        
        print(f"  📋 Reference labels: {list(ref_label_map.values())}")
        print(f"  🔍 OCR detected labels: {[l['text'] for l in ocr_labels]}")
        
        normalized_ocr_labels = []
        for ocr_label in ocr_labels:
            ocr_text = ocr_label['text'].strip()
            matched_ref_label = None
            
            if ocr_text.lower() in ref_label_map:
                matched_ref_label = ref_label_map[ocr_text.lower()]
            else:
                matches = []
                for ref_key, ref_value in ref_label_map.items():
                    if (ocr_text.lower() in ref_key or ref_key in ocr_text.lower()):
                        matches.append(ref_value)
                
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
                normalized_ocr_labels.append({
                    **ocr_label,
                    'normalized_text': ocr_text,
                    'is_valid_ref': False
                })
        
        possible_matches = []
        
        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            """Calculate minimum distance from point to line segment"""
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)

            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        for i, line in enumerate(extracted_lines):
            x1, y1 = line["start"]["x"], line["start"]["y"]
            x2, y2 = line["end"]["x"], line["end"]["y"]
            
            for j, label in enumerate(normalized_ocr_labels):
                if not label['is_valid_ref']:
                    continue
                    
                label_x, label_y = label["center"]
                distance = point_to_segment_distance(label_x, label_y, x1, y1, x2, y2)
                
                possible_matches.append({
                    'line_idx': i,
                    'label_idx': j,
                    'distance': distance,
                    'label_text': label['normalized_text']
                })
        
        possible_matches.sort(key=lambda x: x['distance'])
        
        matched_line_indices = set()
        matched_labels = set()
        
        for line in extracted_lines:
            line['label'] = ''
            
        print("\n  🔗 Assigning labels (greedy unique matching):")
        for match in possible_matches:
            line_idx = match['line_idx']
            label_text = match['label_text']
            
            if line_idx in matched_line_indices or label_text in matched_labels:
                continue
                
            extracted_lines[line_idx]['label'] = label_text
            matched_line_indices.add(line_idx)
            matched_labels.add(label_text)
            print(f"    ✓ Line {line_idx+1} matched to '{label_text}' (dist: {match['distance']:.3f})")
            
        print(f"✓ Matched {len(matched_line_indices)} lines to unique labels")
        
        if reference_lines:
            unmatched_line_indices = [i for i, line in enumerate(extracted_lines) 
                                    if not line.get('label', '')]
            unmatched_ref_labels = [label for label in ref_label_map.values() 
                                   if label not in matched_labels]
            
            if unmatched_line_indices and unmatched_ref_labels:
                print(f"\n  🔄 Assigning unmatched labels to unmatched lines...")
                print(f"    Unmatched lines: {len(unmatched_line_indices)}")
                print(f"    Unmatched labels: {unmatched_ref_labels}")
                
                ref_line_positions = {}
                for ref_line in reference_lines:
                    label = ref_line.get('label', '')
                    if label and label in unmatched_ref_labels:
                        start = ref_line.get('start', {})
                        end = ref_line.get('end', {})
                        if start and end:
                            mid_x = (start.get('x', 0) + end.get('x', 0)) / 2
                            mid_y = (start.get('y', 0) + end.get('y', 0)) / 2
                            ref_line_positions[label] = (mid_x, mid_y)
                
                unmatched_matches = []
                for line_idx in unmatched_line_indices:
                    line = extracted_lines[line_idx]
                    line_mid_x = (line["start"]["x"] + line["end"]["x"]) / 2
                    line_mid_y = (line["start"]["y"] + line["end"]["y"]) / 2
                    
                    for label in unmatched_ref_labels:
                        if label in ref_line_positions:
                            ref_mid_x, ref_mid_y = ref_line_positions[label]
                            distance = np.sqrt((line_mid_x - ref_mid_x)**2 + (line_mid_y - ref_mid_y)**2)
                            unmatched_matches.append({
                                'line_idx': line_idx,
                                'label': label,
                                'distance': distance
                            })
                        else:
                            unmatched_matches.append({
                                'line_idx': line_idx,
                                'label': label,
                                'distance': 999.0
                            })
                
                unmatched_matches.sort(key=lambda x: x['distance'])
                
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
                
                remaining_unmatched_lines = [i for i in unmatched_line_indices if i not in used_line_indices]
                remaining_unmatched_labels = [l for l in unmatched_ref_labels if l not in used_labels]
                
                if remaining_unmatched_lines and remaining_unmatched_labels:
                    print(f"    🔄 Assigning remaining {len(remaining_unmatched_labels)} labels to {len(remaining_unmatched_lines)} lines...")
                    for i, line_idx in enumerate(remaining_unmatched_lines):
                        if i < len(remaining_unmatched_labels):
                            label = remaining_unmatched_labels[i]
                            extracted_lines[line_idx]['label'] = label
                            print(f"    ✓ Assigned remaining label '{label}' to line {line_idx+1}")
        
        if reference_lines:
            lines_without_labels = [i for i, line in enumerate(extracted_lines) if not line.get('label', '')]
            if lines_without_labels:
                print(f"  ⚠️  Warning: {len(lines_without_labels)} lines still without labels after assignment")
            else:
                print(f"  ✓ All lines have been assigned labels")
        
        return extracted_lines


