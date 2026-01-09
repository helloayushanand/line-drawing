import os
from typing import Optional

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3-pro-image-preview"  # Gemini 3 image generation model
    SEEDREAM_API_KEY: str = os.getenv("SEEDREAM_API_KEY", "")
    SEEDREAM_MODEL: str = "seedream-4-0-250828"
    SEEDREAM_API_URL: str = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    BFL_API_KEY: str = os.getenv("BFL_API_KEY", "")
    FLUX_API_URL: str = "https://api.bfl.ai/v1/flux-2-pro"
    MAX_IMAGE_SIZE: int = 2048  # Higher resolution for accuracy
    EDGE_DETECTION_LOW_THRESHOLD: int = 50
    EDGE_DETECTION_HIGH_THRESHOLD: int = 150
    HOUGH_LINE_THRESHOLD: int = 100
    MIN_LINE_LENGTH: int = 50
    MAX_LINE_GAP: int = 10
    SNAP_TO_EDGE_THRESHOLD: int = 20  # pixels
    FEATURE_MATCH_THRESHOLD: float = 0.1  # normalized distance threshold
    LINE_SIMILARITY_DISTANCE_THRESHOLD: float = 0.05  # normalized distance (0-1) between line midpoints to be considered similar
    LINE_SIMILARITY_ANGLE_THRESHOLD: float = 15.0  # degrees - max angle difference to be considered similar
    LINE_OVERLAP_THRESHOLD: float = 0.7  # minimum overlap ratio to consider lines as duplicates
    MEASUREMENT_LINE_MIN_LENGTH_RATIO: float = 0.05  # 5% of image dimension
    MEASUREMENT_LINE_CONTRAST_THRESHOLD: int = 70  # max average pixel value for black lines
    MEASUREMENT_LINE_CONTRAST_MAX: int = 100  # max pixel value allowed in line
    MEASUREMENT_LINE_EDGE_SEPARATION: int = 1  # min pixels from edge for outside lines
    MEASUREMENT_LINE_POSITION_THRESHOLD: float = 0.6  # 70% threshold for inside/outside classification
    BLACK_LINE_THRESHOLD: int = 100  # pixel value threshold for black lines
    LINE_MORPH_KERNEL_SIZE: int = 3  # kernel size for morphological operations
    USE_REFERENCE_BASED_MATCHING: bool = True  # Set to True to use reference-based line matching instead of current logic
    USE_SIMPLE_HOUGH: bool = True  # Set to True to use simple Hough transform (for non-black products)
    # Line extension parameters
    LINE_EXTENSION_STEP_SIZE: int = 2  # pixels to step when extending (smaller = more precise but slower)
    LINE_EXTENSION_MAX_DISTANCE: int = 200  # max pixels to extend in each direction (safety limit)
    LINE_EXTENSION_MIN_PIXEL_RATIO: float = 0.3  # minimum ratio of line pixels in a window to continue extension
    LINE_EXTENSION_WINDOW_SIZE: int = 5  # size of window to check for line pixels when extending

settings = Settings()

