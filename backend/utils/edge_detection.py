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

def snap_point_to_edge(
    point: Tuple[float, float],
    edges: np.ndarray,
    threshold: int = None
) -> Tuple[float, float]:
    """
    Snap a point to the nearest edge pixel
    """
    if threshold is None:
        threshold = settings.SNAP_TO_EDGE_THRESHOLD
    
    x, y = int(point[0]), int(point[1])
    h, w = edges.shape
    
    # Clamp coordinates to image bounds
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    
    # Search in a square around the point
    best_dist = float('inf')
    best_point = (x, y)
    
    for dy in range(-threshold, threshold + 1):
        for dx in range(-threshold, threshold + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if edges[ny, nx] > 0:  # Edge pixel found
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = (nx, ny)
    
    return best_point

def refine_line_with_edges(
    start: Tuple[float, float],
    end: Tuple[float, float],
    edges: np.ndarray
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Refine both endpoints of a line by snapping to edges"""
    refined_start = snap_point_to_edge(start, edges)
    refined_end = snap_point_to_edge(end, edges)
    return refined_start, refined_end

def detect_lines_by_color(
    image: np.ndarray,
    product_mask: Optional[np.ndarray] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Detect black measurement lines using color-based thresholding.
    Product-aware: excludes black pixels that are part of the product.
    
    Args:
        image: Input image (RGB or grayscale)
        product_mask: Optional binary mask where 255 = product, 0 = background
    
    Returns:
        List of lines as (x1, y1, x2, y2) tuples
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    # Threshold to extract black pixels
    # Pixels < threshold are considered black (measurement lines)
    _, binary = cv2.threshold(gray, settings.BLACK_LINE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # Now binary: 255 = black pixels, 0 = white/light pixels
    
    # Apply product mask exclusion if provided
    if product_mask is not None:
        # Ensure mask is same size
        if product_mask.shape[:2] != (h, w):
            product_mask = cv2.resize(product_mask, (w, h))
        
        # Convert mask to single channel if needed
        if len(product_mask.shape) > 2:
            mask_gray = cv2.cvtColor(product_mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = product_mask
        
        # Strategy: Only exclude black pixels that are exactly on product edges
        # This allows measurement lines inside the product (like between legs or on surfaces) to be detected
        # We identify edge pixels by finding the boundary of the product mask
        kernel_edge = np.ones((3, 3), np.uint8)
        # Dilate and erode to find boundary pixels
        dilated_mask = cv2.dilate(mask_gray, kernel_edge, iterations=1)
        eroded_mask = cv2.erode(mask_gray, kernel_edge, iterations=1)
        # Edge pixels are those in dilated but not in eroded mask (the boundary)
        edge_pixels = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(eroded_mask))
        
        # Exclude black pixels that are on product edges (to avoid detecting black product edges)
        # But keep black pixels that are inside the product (not on edge) - these are measurement lines
        binary[edge_pixels > 0] = 0
        
        # For outside lines, exclude pixels very close to product edges using distance transform
        # Calculate distance transform from product edge
        inverted_mask = cv2.bitwise_not(mask_gray)
        dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
        
        # Only exclude pixels outside the product that are too close to edges
        # This prevents detecting product edges when product is black
        # But keep pixels inside the product (they're already handled by edge_pixels exclusion)
        outside_mask = (mask_gray == 0)
        too_close_outside = (dist_transform < settings.MEASUREMENT_LINE_EDGE_SEPARATION) & outside_mask
        binary[too_close_outside] = 0
    
    # Apply morphological closing to connect broken line segments
    kernel = np.ones((settings.LINE_MORPH_KERNEL_SIZE, settings.LINE_MORPH_KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Use Hough Transform directly on the binary mask (not on Canny edges)
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

