"""
Direct test script for Line Detection (no server required)
Usage: python test_direct.py <input_image> <reference_image> [product_type] [expected_line_count] [line_types...]
"""
import sys
import base64
import io
from PIL import Image
from services.line_detector import LineDetector

def encode_image_file(image_path: str) -> str:
    """Encode image file to base64 string"""
    from PIL import Image
    import base64
    import io
    
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_direct.py <input_image> <reference_image> [product_type] [expected_line_count] [line_types...]")
        print("\nExample:")
        print("  python test_direct.py product.jpg reference.jpg chair 3 Width Height Depth")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    reference_image_path = sys.argv[2]
    product_type = sys.argv[3] if len(sys.argv) > 3 else None
    expected_line_count = int(sys.argv[4]) if len(sys.argv) > 4 else None
    line_types = sys.argv[5:] if len(sys.argv) > 5 else None
    
    print("="*60)
    print("Line Detection Test (Direct Mode)")
    print("="*60)
    print(f"Input image: {input_image_path}")
    print(f"Reference image: {reference_image_path}")
    if product_type:
        print(f"Product type: {product_type}")
    if expected_line_count:
        print(f"Expected line count: {expected_line_count}")
    if line_types:
        print(f"Line types: {', '.join(line_types)}")
    print()
    
    try:
        # Encode images
        print("Encoding images...")
        input_img_b64 = encode_image_file(input_image_path)
        ref_img_b64 = encode_image_file(reference_image_path)
        print("✓ Images encoded")
        
        # Initialize detector
        print("Initializing line detector...")
        detector = LineDetector()
        print("✓ Line detector initialized")
        
        # Detect lines
        print("\nDetecting lines...")
        print("This may take a moment...")
        lines, method_used, confidence = detector.detect_lines(
            input_image_base64=f"data:image/png;base64,{input_img_b64}",
            reference_image_base64=f"data:image/png;base64,{ref_img_b64}",
            product_type=product_type,
            expected_line_count=expected_line_count,
            line_types=line_types
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Method used: {method_used}")
        print(f"Confidence score: {confidence:.2f}")
        print(f"Number of lines detected: {len(lines)}")
        
        if lines:
            # Show summary of lines with labels
            print("\nLines Summary:")
            print("-" * 60)
            for i, line in enumerate(lines, 1):
                label = line.label if line.label else f"Line {i}"
                print(f"  {i}. {label}")
            
            # Print JSON format (compact)
            print("\n" + "="*60)
            print("JSON Output:")
            print("-" * 60)
            import json
            json_output = {
                "lines": [
                    {
                        "start": {"x": line.start.x, "y": line.start.y},
                        "end": {"x": line.end.x, "y": line.end.y},
                        "label": line.label,
                        "confidence": line.confidence
                    }
                    for line in lines
                ],
                "method_used": method_used,
                "confidence_score": confidence
            }
            print(json.dumps(json_output, indent=2))
        else:
            print("\n⚠️  No lines detected!")
            print("Possible reasons:")
            print("  - Images may not be similar enough")
            print("  - Gemini API key may not be set")
            print("  - Check that images are on white backgrounds")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Image file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

