"""
Test script for extracting lines from already-generated images
Usage: python test_extraction.py <generated_image> [expected_line_count]
"""
import sys
import json
from PIL import Image, ImageDraw
from services.coordinate_extractor import CoordinateExtractor
from models.schemas import LineDrawing, Point

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <generated_image> [expected_line_count]")
        print("\nExample:")
        print("  python test_extraction.py generated_images/generated_123.png 3")
        sys.exit(1)
    
    generated_image_path = sys.argv[1]
    expected_line_count = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print("="*60)
    print("Line Extraction Test (CV Only - Generated Image)")
    print("="*60)
    print(f"Generated image: {generated_image_path}")
    if expected_line_count:
        print(f"Expected line count: {expected_line_count}")
    print()
    
    try:
        # Load generated image only
        print("Loading generated image...")
        generated_image = Image.open(generated_image_path)
        print(f"✓ Generated image size: {generated_image.size}")
        print()
        
        # Initialize extractor
        print("Initializing coordinate extractor...")
        extractor = CoordinateExtractor()
        print("✓ Extractor initialized")
        print()
        
        # Extract lines (only from generated image)
        print("Extracting lines from generated image...")
        print("Using CV methods only (no API calls, no original image needed)...")
        if expected_line_count:
            print(f"Filtering to top {expected_line_count} lines outside product...")
        extracted_lines = extractor.extract_lines_from_generated_image_only(
            generated_image,
            expected_line_count=expected_line_count
        )
        
        print(f"✓ Extracted {len(extracted_lines)} lines")
        print()
        
        # Convert to LineDrawing objects for display
        lines = []
        for line in extracted_lines:
            lines.append(LineDrawing(
                start=Point(x=line.get("start", {}).get("x", 0), y=line.get("start", {}).get("y", 0)),
                end=Point(x=line.get("end", {}).get("x", 0), y=line.get("end", {}).get("y", 0)),
                label=line.get("label", ""),
                confidence=line.get("confidence")
            ))
        
        # Visualize lines on generated image
        print("\n📊 Visualizing detected lines on generated image...")
        viz_image = generated_image.copy()
        draw = ImageDraw.Draw(viz_image)
        w, h = viz_image.size
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            start_x = int(line.start.x * w)
            start_y = int(line.start.y * h)
            end_x = int(line.end.x * w)
            end_y = int(line.end.y * h)
            
            # Draw line (thicker to be visible)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=5)
            # Draw endpoints
            draw.ellipse([start_x-8, start_y-8, start_x+8, start_y+8], fill=color, outline=(255, 255, 255), width=2)
            draw.ellipse([end_x-8, end_y-8, end_x+8, end_y+8], fill=color, outline=(255, 255, 255), width=2)
            # Draw label
            label_text = line.label if line.label else f"Line {i+1}"
            if line.confidence:
                label_text += f" ({line.confidence:.2f})"
            draw.text((start_x + 10, start_y - 15), label_text, fill=color)
        
        # Save visualization
        output_path = "detected_lines_visualization.png"
        viz_image.save(output_path)
        print(f"✓ Saved visualization to: {output_path}")
        print("   Open this image to verify if lines are correct!")
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Number of lines detected: {len(lines)}")
        
        if lines:
            print("\nDetected Lines:")
            print("-" * 60)
            for i, line in enumerate(lines, 1):
                label = line.label if line.label else f"Line {i}"
                conf_str = f" (confidence: {line.confidence:.3f})" if line.confidence else ""
                print(f"  {i}. {label}{conf_str}")
                print(f"     Start: ({line.start.x:.4f}, {line.start.y:.4f})")
                print(f"     End:   ({line.end.x:.4f}, {line.end.y:.4f})")
            
            # Print JSON format
            print("\n" + "="*60)
            print("JSON Output:")
            print("-" * 60)
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
                "method_used": "cv_extraction",
                "total_lines": len(lines)
            }
            print(json.dumps(json_output, indent=2))
        else:
            print("\n⚠️  No lines detected!")
            print("Possible reasons:")
            print("  - Generated image might not have visible lines")
            print("  - Lines might be too faint")
            print("  - Product mask detection might be filtering out all lines")
        
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
