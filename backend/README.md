# Line Detection Backend API

A Python backend service for detecting measurement lines on product images using AI and computer vision techniques.

## Features

- **Multiple Detection Methods**: 
  - Image generation with Gemini 3 + coordinate extraction
  - Feature matching for similar products
  - Direct AI coordinate prediction
- **High Accuracy**: Edge detection refinement and multi-method consensus
- **Similar Product Support**: Feature matching works well for products in the same category
- **RESTful API**: FastAPI-based endpoint for easy integration

## Setup

### Prerequisites

- Python 3.8+
- Gemini API key

### Installation

1. Create a virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

## Configuration

Edit `config.py` to adjust:
- Model name (update `GEMINI_MODEL` with actual Gemini 3 model name)
- Image processing parameters
- Edge detection thresholds
- Feature matching settings

## Running the Server

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Usage

### Endpoint: `POST /detect-lines`

**Request Body:**
```json
{
  "input_image_base64": "data:image/png;base64,iVBORw0KG...",
  "reference_image_base64": "data:image/png;base64,iVBORw0KG...",
  "product_type": "chair",
  "expected_line_count": 3,
  "line_types": ["Width", "Height", "Depth"]
}
```

**Response:**
```json
{
  "lines": [
    {
      "start": {"x": 0.1, "y": 0.2},
      "end": {"x": 0.9, "y": 0.2},
      "label": "Width",
      "confidence": 0.95
    }
  ],
  "method_used": "image_generation",
  "processing_time": 2.34,
  "confidence_score": 0.92
}
```

### Example with cURL

```bash
curl -X POST "http://localhost:8000/detect-lines" \
  -H "Content-Type: application/json" \
  -d '{
    "input_image_base64": "...",
    "reference_image_base64": "...",
    "product_type": "chair",
    "expected_line_count": 3,
    "line_types": ["Width", "Height", "Depth"]
  }'
```

### Example with Python

```python
import requests
import base64

# Read images
with open("product.jpg", "rb") as f:
    input_img = base64.b64encode(f.read()).decode()

with open("reference.jpg", "rb") as f:
    ref_img = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/detect-lines",
    json={
        "input_image_base64": f"data:image/jpeg;base64,{input_img}",
        "reference_image_base64": f"data:image/jpeg;base64,{ref_img}",
        "product_type": "chair",
        "expected_line_count": 3,
        "line_types": ["Width", "Height", "Depth"]
    }
)

result = response.json()
print(f"Found {len(result['lines'])} lines")
print(f"Method: {result['method_used']}")
print(f"Confidence: {result['confidence_score']}")
```

## Architecture

### Services

- **GeminiService**: Handles Gemini API calls for image generation and direct prediction
- **CoordinateExtractor**: Extracts line coordinates from generated images using multiple methods
- **FeatureMatcher**: Matches features between similar products for alignment
- **LineDetector**: Main orchestrator that tries multiple methods and returns best results

### Detection Methods

1. **Image Generation** (Preferred):
   - Uses Gemini 3 to generate image with lines
   - Extracts coordinates via image diff, AI extraction, and CV
   - Most accurate when available

2. **Feature Matching**:
   - Detects features in both images
   - Calculates homography transformation
   - Transforms reference lines to input image space
   - Good for similar products

3. **Direct Prediction** (Fallback):
   - AI directly predicts coordinates
   - Refined with edge detection
   - Works when other methods fail

### Post-Processing

All detected lines are refined by:
- Snapping endpoints to nearest edges
- Validating geometric constraints
- Calculating confidence scores

## Notes

- **Gemini 3 Model Name**: Update `GEMINI_MODEL` in `config.py` with the actual model name when available
- **Image Generation**: The image generation method may need adjustment based on Gemini 3 API documentation
- **Base64 Format**: Images should be base64 encoded, with or without data URI prefix
- **Coordinates**: All coordinates are normalized to 0-1 range (x, y)

## Troubleshooting

1. **API Key Error**: Ensure `GEMINI_API_KEY` is set correctly
2. **No Lines Detected**: Try adjusting thresholds in `config.py`
3. **Low Confidence**: Check image quality and ensure products are similar
4. **Import Errors**: Ensure all dependencies are installed

## License

Same as parent project (Apache 2.0)

