from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models.schemas import LineDetectionRequest, LineDetectionResponse, LineDrawing
from services.line_detector import LineDetector
import time
import os

app = FastAPI(title="Line Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router with /api prefix
api_router = APIRouter(prefix="/api")

# Initialize line detector
line_detector = LineDetector()

# Get the path to the dist folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "dist")

# Serve static files
if os.path.exists(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

@api_router.post("/detect-lines", response_model=LineDetectionResponse)
async def detect_lines(request: LineDetectionRequest):
    """
    Main endpoint for line detection
    
    Input:
    - input_image_base64: Base64 encoded product image (white background)
    - reference_image_base64: Base64 encoded reference image with lines
    - product_type: Optional product category (e.g., "chair", "table")
    - expected_line_count: Optional expected number of lines
    - line_types: Optional list of line labels (e.g., ["Width", "Height", "Depth"])
    - model: Optional model selection ("gemini" or "flux")
    - reference_lines: Optional list of reference lines with labels
    
    Output:
    - lines: List of detected lines with coordinates and labels
    - method_used: Which detection method was used
    - processing_time: Time taken in seconds
    - confidence_score: Confidence in the results (0-1)
    """
    try:
        start_time = time.time()
        
        # Validate inputs
        if not request.input_image_base64:
            raise HTTPException(status_code=400, detail="input_image_base64 is required")
        if not request.reference_image_base64:
            raise HTTPException(status_code=400, detail="reference_image_base64 is required")
        
        # Detect lines
        lines, method_used, confidence, generated_image_base64, input_image_base64 = line_detector.detect_lines(
            input_image_base64=request.input_image_base64,
            reference_image_base64=request.reference_image_base64,
            product_type=request.product_type,
            expected_line_count=request.expected_line_count,
            line_types=request.line_types,
            model=request.model or "gemini",
            reference_lines=[line.dict() for line in request.reference_lines] if request.reference_lines else None,
        )
        
        processing_time = time.time() - start_time
        
        return LineDetectionResponse(
            lines=lines,
            method_used=method_used,
            processing_time=processing_time,
            confidence_score=confidence,
            generated_image_base64=generated_image_base64,
            input_image_base64=input_image_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Include API router
app.include_router(api_router)

# Keep /docs and /openapi.json accessible at root
@app.get("/docs")
async def docs_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/docs")

# Serve frontend - must be last
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve frontend for all routes except API endpoints"""
    if full_path.startswith(("api", "docs", "openapi.json")):
        raise HTTPException(status_code=404, detail="Not found")
    
    index_path = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
