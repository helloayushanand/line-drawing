from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image
from typing import List, Optional
from utils.image_utils import pil_to_numpy


class OCRService:
    def __init__(self):
        self.engine = RapidOCR()
    
    def _preprocess_for_ocr(self, image: Image.Image) -> np.ndarray:
        img_arr = pil_to_numpy(image)
        return img_arr
    
    def extract_labels(
        self, 
        image: Image.Image,
        expected_labels: Optional[List[str]] = None
    ) -> List[dict]:
        preprocessed = self._preprocess_for_ocr(image)
        h, w = preprocessed.shape[:2]
        
        results, _ = self.engine(preprocessed)
        if not results:
            results = []
        
        labels = []
        for bbox, text, confidence in results:
            try:
                conf_val = float(confidence)
            except (ValueError, TypeError):
                conf_val = 0.0

            if conf_val < 0.3:
                continue
            
            if expected_labels:
                text_lower = text.lower().strip()
                matches = any(expected.lower() in text_lower or text_lower in expected.lower() 
                             for expected in expected_labels)
                if not matches:
                    continue
            
            bbox_array = np.array(bbox)
            center_x = float(np.mean(bbox_array[:, 0]) / w)
            center_y = float(np.mean(bbox_array[:, 1]) / h)
            
            safe_bbox = [[float(p[0]), float(p[1])] for p in bbox]
            
            labels.append({
                "text": str(text).strip(),
                "bbox": safe_bbox,
                "center": (center_x, center_y),
                "confidence": float(conf_val)
            })
        
        return labels
    
    def match_lines_to_labels(
        self,
        extracted_lines: List[dict],
        ocr_labels: List[dict],
        reference_lines: Optional[List[dict]] = None
    ) -> List[dict]:
        if not ocr_labels:
            return extracted_lines
        
        ref_label_map = {}
        if reference_lines:
            for ref_line in reference_lines:
                label = ref_line.get('label', '')
                if label:
                    ref_label_map[label.lower().strip()] = label
        
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
        
        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                return np.sqrt((px - x1)**2 + (py - y1)**2)

            t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

        possible_matches = []
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
            
        for match in possible_matches:
            line_idx = match['line_idx']
            label_text = match['label_text']
            
            if line_idx in matched_line_indices or label_text in matched_labels:
                continue
                
            extracted_lines[line_idx]['label'] = label_text
            matched_line_indices.add(line_idx)
            matched_labels.add(label_text)
        
        if reference_lines:
            unmatched_line_indices = [i for i, line in enumerate(extracted_lines) 
                                    if not line.get('label', '')]
            unmatched_ref_labels = [label for label in ref_label_map.values() 
                                   if label not in matched_labels]
            
            if unmatched_line_indices and unmatched_ref_labels:
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
                
                remaining_unmatched_lines = [i for i in unmatched_line_indices if i not in used_line_indices]
                remaining_unmatched_labels = [l for l in unmatched_ref_labels if l not in used_labels]
                
                if remaining_unmatched_lines and remaining_unmatched_labels:
                    for i, line_idx in enumerate(remaining_unmatched_lines):
                        if i < len(remaining_unmatched_labels):
                            label = remaining_unmatched_labels[i]
                            extracted_lines[line_idx]['label'] = label
        
        if reference_lines:
            ref_labels = [line.get('label', '') for line in reference_lines if line.get('label', '')]
            lines_without_labels = [i for i, line in enumerate(extracted_lines) if not line.get('label', '')]
            
            if lines_without_labels and ref_labels:
                used_labels = {line.get('label', '') for line in extracted_lines if line.get('label', '')}
                available_labels = [label for label in ref_labels if label not in used_labels]
                
                for i, line_idx in enumerate(lines_without_labels):
                    if i < len(available_labels):
                        extracted_lines[line_idx]['label'] = available_labels[i]
                    elif ref_labels:
                        extracted_lines[line_idx]['label'] = ref_labels[i % len(ref_labels)]
        
        return extracted_lines


