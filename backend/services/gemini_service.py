from google import genai
from google.genai import types
from PIL import Image
import io
from typing import List, Optional
from config import settings
from utils.image_utils import decode_base64_image

class GeminiService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    def generate_image_with_lines(
        self, 
        input_image_base64: str, 
        reference_image_base64: str,
        product_type: Optional[str] = None,
        line_types: Optional[List[str]] = None,
        expected_line_count: Optional[int] = None,
        reference_lines: Optional[List[dict]] = None
    ) -> Optional[Image.Image]:
        """Generate image with measurement lines using Gemini 3"""
        try:
            input_img = decode_base64_image(input_image_base64)
            ref_img = decode_base64_image(reference_image_base64)
            
            label_info = ""
            if reference_lines:
                label_info = "\n\n====================================================================\n"
                label_info += "REFERENCE LINE LABELS (MUST PRESERVE IN OUTPUT):\n"
                label_info += "====================================================================\n\n"
                for i, line in enumerate(reference_lines, 1):
                    label = line.get('label', f'L{i}')
                    label_info += f"- Line {i}: \"{label}\"\n"
                label_info += "\n**CRITICAL**: You MUST draw these exact labels (e.g., 'L1', 'L2') on the corresponding lines.\n"
                label_info += "Place each label near its line (preferably near the start point) in a readable font.\n"
                label_info += "Labels must be clearly visible in black text.\n\n"
            
            line_info = ""
            if line_types:
                line_info = f" The reference shows {len(line_types)} lines measuring: {', '.join(line_types)}."
            
            count_instruction = ""
            if expected_line_count:
                count_instruction = f"4. **COUNT**: You must draw **EXACTLY {expected_line_count}** lines. This is a STRICT REQUIREMENT."
            else:
                count_instruction = "4. **COUNT**: You must draw **EXACTLY** the same number of lines as in the reference."

            
            product_info = f"Product type: {product_type}. " if product_type else ""
            
            prompt = f"""

You are a TECHNICAL MEASUREMENT SCHEMATIC GENERATOR.

====================================================================
IMPORTANT: IMAGE FORMAT AND LABELS
====================================================================

Both images you receive are:
1. **WATERMARKED** - Each image has a clear label at the bottom:
   - The FIRST image is labeled "INPUT IMAGE" at the bottom (this image has a faded, desaturated, grey/black appearance)
   - The SECOND image is labeled "REFERENCE IMAGE" at the bottom (this image retains its original colors)

2. **INPUT IMAGE APPEARANCE**:
   - The FIRST image (labeled "INPUT IMAGE") has been processed to appear faded, desaturated, and in grey/black tones
   - It may look ghost-like, translucent, or washed-out - this is intentional
   - It retains subtle hints of the original color but is primarily grey/black

3. **REFERENCE IMAGE APPEARANCE**:
   - The SECOND image (labeled "REFERENCE IMAGE") retains its original colors
   - It is NOT greyscale or faded - it shows the reference with full color

**CRITICAL**: You MUST draw lines ONLY on the image labeled "INPUT IMAGE" (the FIRST image).
You MUST NOT draw lines on the image labeled "REFERENCE IMAGE" (the SECOND image).

The "REFERENCE IMAGE" is ONLY for reference - to see which measurement lines exist.
The "INPUT IMAGE" is where you must draw the lines. 
====================================================================
YOUR TASK
====================================================================

Your task is to add measurement guide lines to the PRODUCT in the FIRST image (labeled "INPUT IMAGE") 
(without rotating the product), using the SECOND image (labeled "REFERENCE IMAGE") strictly as a 
REFERENCE for which measurement lines exist.

{label_info}

====================================================================
🚨 RULE #1: ABSOLUTELY NO ROTATION - CRITICAL FAILURE IF VIOLATED 🚨
====================================================================

⚠️ THIS IS THE MOST IMPORTANT RULE - READ THIS FIRST ⚠️

ABSOLUTELY FORBIDDEN:
- Changing the input product's orientation, angle, or rotation
- Aligning the product to match the SECOND (reference) image's orientation
- Any transformation that changes how the product appears in the input image

MANDATORY REQUIREMENT:
- The product in your output MUST be IDENTICAL in orientation, angle, size, and position to the product in the FIRST (input) image
- You must TRACE or COPY the product from the input image exactly as it appears
- The product outline should be a pixel-perfect or near-pixel-perfect match to the input product


====================================================================
STEP 0 — PRE-PROCESSING: ORIENTATION CHECK (MANDATORY FIRST STEP)
====================================================================

BEFORE doing anything else, you MUST perform this check:

1. **EXAMINE THE FIRST IMAGE (INPUT)** - Look for the "INPUT IMAGE" watermark at the bottom:
   - This is the image on which you MUST draw the lines
   - Look at the product's orientation, angle, and rotation

2. **EXAMINE THE SECOND IMAGE (REFERENCE)** - Look for the "REFERENCE IMAGE" watermark at the bottom:
   - This image is ONLY for reference - do NOT draw lines on it

====================================================================
CRITICAL OBJECTIVE:  LINE TRANSFER (NO EXCEPTIONS)
====================================================================

You must draw the lines on the image labeled "INPUT IMAGE" (in black color), NOT on the image labeled "REFERENCE IMAGE". The lines must be drawn representing the same measurements as they are representing in the reference image.
You are NOT allowed to invent, infer, split, merge, extend, or add measurements.

Only the measurement lines are drawn, and they must adapt to match how the physical features 
appear in the input image's orientation. Adjust them according to the input image dimensions. 

The number of measurement lines must be same as the number of measurement lines in reference image.

====================================================================
VISUAL OUTPUT REQUIREMENTS (HARD CONSTRAINTS)
====================================================================

BACKGROUND:
- Pure white only (RGB 255, 255, 255)

MEASUREMENT LINES:
- Color: Pure black (#000000) - ABSOLUTELY NO COLORS, ONLY BLACK/GREY
- Style: Solid lines with dot endpoints
- Every black line MUST represent a real measurement from the reference
- **CRITICAL**: The input image is faded/grey, so your output lines MUST be pure black (#000000) for clear visibility
- **FORBIDDEN**: Do NOT use any colors (red, blue, green, etc.) - ONLY use black (#000000)
- Draw pure black lines on the white background - they should stand out clearly against the faded input image

====================================================================
NEGATIVE CONSTRAINTS (ABSOLUTE)
====================================================================

- DO NOT draw extra lines. 
- DO NOT invent or guess measurements
- DO NOT reposition or resize the product. You can not change the position or size of the product.
- 🚨 DO NOT rotate the product - THIS IS ABSOLUTELY FORBIDDEN 🚨
- 🚨 DO NOT change the product's orientation, angle, or rotation - CRITICAL FAILURE 🚨
- DO NOT use reference image angles for lines - lines must use input image angles
- DO NOT draw lines at reference angles when orientations differ - adapt to input angles
- DO identify physical features and draw lines connecting them at their actual angles in the input image 
- You are not allowed to change the input product in any way. You can only draw lines on or around the product.
- The product details must not change. Keeping the same product in the final image is of utmost importance.
- You must not enlarge the input product. It should be the same size and at same location as in the input image.
- You must not rotate the input product. It should be the same orientation as in the input image.
- Do not keep 'INPUT IMAGE' text in the final image.
- Do not enlarge the input product, if it is in a small area, then generate the image keeping the product in the same area. You must not bring it to center and enlarge it.


====================================================================
LINE LOCK RULE (CRITICAL)
====================================================================

- You may ONLY draw lines listed in the registry (L1…LN). One line for each L.
- You may NOT split one line into multiple lines
- You may NOT combine lines
- You may NOT replace missing lines with new ones
- You must label the correct set of lines in the generated image as in reference image. Do not put wrong labels.
- The lines must compliment the edges of the product in the input image. Do not just copy paste the location of lines from reference image. Their job is to represent the dimensions of the input product. 
- The job is to find the similar relevant dimension in the input product and draw a line representing the same dimension. If it is impossible to find the similar dimension then draw the line representing the closest dimension.



====================================================================
ORIENTATION HANDLING (CRITICAL FOR DIFFERENT ORIENTATIONS)
====================================================================

When the reference and input images have DIFFERENT orientations:

CORRECT APPROACH:
1. Identify the physical feature in the reference (e.g., "horizontal width measurement")
2. Locate that SAME physical feature in the input image
3. Observe how that feature appears in the input image (it may be rotated/angled differently)
4. Draw the line connecting the feature points using the ACTUAL angle from the input image
5. The line should follow the input image's coordinate system
====================================================================
STEP 3 — SCHEMATIC DRAWING
====================================================================

BEFORE DRAWING:
1. Identify the image labeled "INPUT IMAGE" at the bottom - this is where you draw
2. Look at the FIRST (input) image's product (the one labeled "INPUT IMAGE")
3. Trace/copy that product EXACTLY as it appears (same orientation, size, position)
4. Draw it as a faint gray outline
5. Verify: Does your product match the input product's orientation? If NO, start over.

THEN DRAW LINES:
- Draw ONLY the registered measurement lines
- Color: pure black (#000000)
- Style: solid with dot endpoints
- Product outline must remain faint gray and minimal
- Lines must use the INPUT IMAGE's coordinate system and angles
- Lines must connect physical features at the angles they appear in the INPUT image
- return the image with drawn lines on input image. do not change the input product in any way.

Compliance with line count is more important than completeness.
"""

 

            # Generate image using new API format (matching working example)
            try:
                print("🔄 Attempting Gemini 3 image generation...")
                
                # Pass images directly in contents, no config needed for image generation model
                response = self.client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=[input_img, ref_img, prompt],
                    config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                    image_config=types.ImageConfig(
                        image_size="2K",
                        aspect_ratio=self._get_closest_aspect_ratio(input_img),
                    ),
                )
                )
                
                print("✓ Received response from Gemini 3")
                print(response, "GEMINI response")
                
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"⚠️  Prompt feedback: {response.prompt_feedback}")
                    if hasattr(response.prompt_feedback, 'block_reason'):
                        print(f"❌ BLOCKED: {response.prompt_feedback.block_reason}")
                    if hasattr(response.prompt_feedback, 'safety_ratings'):
                        print(f"⚠️  Safety ratings: {response.prompt_feedback.safety_ratings}")
                
                generated_image = None
                generation_text = None
                
                for candidate in response.candidates:
                    print(f"  Candidate {candidate.index}: finish_reason={candidate.finish_reason}")
                    if candidate.content is None:
                        print(f"  ❌ Candidate {candidate.index} has None content")
                        continue
                    if candidate.content.parts is None:
                        print(f"  ❌ Candidate {candidate.index} has None parts")
                        continue
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text is not None:
                            generation_text = part.text
                            print(f"Response text: {generation_text}")
                        elif hasattr(part, "inline_data") and part.inline_data is not None:
                            print("✓ Inline data found, extracting image...")
                            generated_image = Image.open(io.BytesIO(part.inline_data.data))
                            print(f"✓ Found image in response: {generated_image.size}")
                            break
                    if generated_image:
                        break
                
                if generated_image:
                    print(f"✓ Successfully generated image: {generated_image.size}")
                    return generated_image
                else:
                    print("❌ No image found in response")
                    if generation_text:
                        print(f"Response was text instead: {generation_text[:200]}...")
                    finish_reasons = [f"candidate {c.index}: {c.finish_reason}" for c in response.candidates]
                    if finish_reasons:
                        print(f"  Finish reasons: {', '.join(finish_reasons)}")
                    return None
                    
            except Exception as e:
                print(f"❌ Image generation failed with error: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"Error in generate_image_with_lines: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_closest_aspect_ratio(self, image: Image.Image) -> str:
        """Return closest supported aspect ratio"""
        w, h = image.size
        ratio = w / h
        
        standards = {
            '1:1': 1.0,
            '3:4': 3/4,
            '4:3': 4/3,
            '9:16': 9/16,
            '16:9': 16/9
        }
        
        closest_ratio = min(standards.keys(), key=lambda k: abs(standards[k] - ratio))
        print(f"  📊 Input aspect ratio: {ratio:.2f} ({w}x{h}), closest standard: {closest_ratio}")
        return closest_ratio

