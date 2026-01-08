import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import settings

def detect_edges(image: np.ndarray) -> np.ndarray:
    """Canny edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(
        gray,
        settings.EDGE_DETECTION_LOW_THRESHOLD,
        settings.EDGE_DETECTION_HIGH_THRESHOLD
    )
    return edges

def detect_lines_hough(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect lines using Hough Transform"""
    edges = detect_edges(image)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=settings.HOUGH_LINE_THRESHOLD,
        minLineLength=settings.MIN_LINE_LENGTH,
        maxLineGap=settings.MAX_LINE_GAP
    )
    
    if lines is None:
        return []
    
    return [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]

def detect_lines_by_color(
    image: np.ndarray,
    product_mask: Optional[np.ndarray] = None
) -> List[Tuple[int, int, int, int]]:
    """Detect black measurement lines using color-based thresholding"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    _, binary = cv2.threshold(gray, settings.BLACK_LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    if product_mask is not None:
        if product_mask.shape[:2] != (h, w):
            product_mask = cv2.resize(product_mask, (w, h))
        
        if len(product_mask.shape) > 2:
            mask_gray = cv2.cvtColor(product_mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = product_mask
        
        kernel_edge = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask_gray, kernel_edge, iterations=1)
        eroded_mask = cv2.erode(mask_gray, kernel_edge, iterations=1)
        edge_pixels = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(eroded_mask))
        binary[edge_pixels > 0] = 0
        
        inverted_mask = cv2.bitwise_not(mask_gray)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        outside_mask = (mask_gray == 0)
        too_close_outside = (dist_transform < settings.MEASUREMENT_LINE_EDGE_SEPARATION) & outside_mask
        binary[too_close_outside] = 0
    
    kernel = np.ones((settings.LINE_MORPH_KERNEL_SIZE, settings.LINE_MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi/180,
        threshold=settings.HOUGH_LINE_THRESHOLD,
        minLineLength=settings.MIN_LINE_LENGTH,
        maxLineGap=settings.MAX_LINE_GAP
    )
    
    if lines is None:
        return []
    
    return [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]

