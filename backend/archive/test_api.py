"""
Simple test script for the Line Detection API
Usage: python test_api.py
"""
import requests
import base64
import json
import sys

def encode_image(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_detect_lines(input_image_path: str, reference_image_path: str, 
                     product_type: str = None, expected_line_count: int = None,
                     line_types: list = None):
    """Test the /detect-lines endpoint"""
    
    # Encode images
    print("Encoding images...")
    input_img_b64 = encode_image(input_image_path)
    ref_img_b64 = encode_image(reference_image_path)
    
    # Prepare request
    payload = {
        "input_image_base64": f"data:image/jpeg;base64,{input_img_b64}",
        "reference_image_base64": f"data:image/jpeg;base64,{ref_img_b64}",
    }
    
    if product_type:
        payload["product_type"] = product_type
    if expected_line_count:
        payload["expected_line_count"] = expected_line_count
    if line_types:
        payload["line_types"] = line_types
    
    # Make request
    print("Sending request to API...")
    try:
        response = requests.post(
            "http://localhost:8000/detect-lines",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Print results
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(f"Method used: {result['method_used']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"Confidence score: {result.get('confidence_score', 0):.2f}")
        print(f"Number of lines detected: {len(result['lines'])}")
        print("\nLines:")
        for i, line in enumerate(result['lines'], 1):
            print(f"  {i}. {line['label']}")
            print(f"     Start: ({line['start']['x']:.3f}, {line['start']['y']:.3f})")
            print(f"     End:   ({line['end']['x']:.3f}, {line['end']['y']:.3f})")
            if line.get('confidence'):
                print(f"     Confidence: {line['confidence']:.2f}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_api.py <input_image> <reference_image> [product_type] [expected_line_count] [line_types...]")
        print("\nExample:")
        print("  python test_api.py product.jpg reference.jpg chair 3 Width Height Depth")
        sys.exit(1)
    
    input_image = sys.argv[1]
    reference_image = sys.argv[2]
    product_type = sys.argv[3] if len(sys.argv) > 3 else None
    expected_line_count = int(sys.argv[4]) if len(sys.argv) > 4 else None
    line_types = sys.argv[5:] if len(sys.argv) > 5 else None
    
    test_detect_lines(
        input_image,
        reference_image,
        product_type,
        expected_line_count,
        line_types
    )

