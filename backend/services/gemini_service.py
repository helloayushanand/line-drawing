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

Your task is to add measurement guide lines to the PRODUCT in the FIRST image (without rotating the product),
using the SECOND image strictly as a REFERENCE for which measurement lines exist.

{label_info}

====================================================================
🚨 RULE #1: ABSOLUTELY NO ROTATION - CRITICAL FAILURE IF VIOLATED 🚨
====================================================================

⚠️ THIS IS THE MOST IMPORTANT RULE - READ THIS FIRST ⚠️

ABSOLUTELY FORBIDDEN:
- Rotating the product from the FIRST (input) image
- Changing the product's orientation, angle, or rotation
- Aligning the product to match the SECOND (reference) image's orientation
- Rotating the product to make it "look better" or "match better"
- Any transformation that changes how the product appears in the input image

MANDATORY REQUIREMENT:
- The product in your output MUST be IDENTICAL in orientation, angle, size, and position to the product in the FIRST (input) image
- You must TRACE or COPY the product from the input image exactly as it appears
- The product outline should be a pixel-perfect or near-pixel-perfect match to the input product
- If the input product is rotated 45°, your output product must also be rotated 45°
- If the input product is horizontal, your output product must also be horizontal
- If the input product is vertical, your output product must also be vertical

IF YOU ROTATE THE PRODUCT, YOU HAVE FAILED THE TASK COMPLETELY.

====================================================================
STEP 0 — PRE-PROCESSING: ORIENTATION CHECK (MANDATORY FIRST STEP)
====================================================================

BEFORE doing anything else, you MUST perform this check:

1. **EXAMINE THE FIRST IMAGE (INPUT)**:
   - Look at the product's orientation, angle, and rotation
   - Note: Is it horizontal? Vertical? Rotated? At what angle?
   - Remember this orientation - it is SACRED and cannot be changed

2. **EXAMINE THE SECOND IMAGE (REFERENCE)**:
   - Look at the product's orientation, angle, and rotation
   - Note: Is it horizontal? Vertical? Rotated? At what angle?

3. **COMPARE ORIENTATIONS**:
   - Are they the same? → Good, proceed normally
   - Are they different? → THIS IS NORMAL AND EXPECTED
   - If different: DO NOT try to align them. DO NOT rotate the input product.
   - If different: The input product orientation is CORRECT. Keep it exactly as is.

4. **SET YOUR MENTAL MODEL**:
   - "The output product MUST match the FIRST image's product orientation exactly"
   - "I will trace/copy the product from the FIRST image, preserving its exact orientation"
   - "I will only add measurement lines, adapting their angles to the input product's orientation"

5. **VERIFICATION BEFORE PROCEEDING**:
   - Ask yourself: "Will I preserve the input product's exact orientation?"
   - If YES → Proceed to next steps
   - If NO → STOP and reconsider. You must preserve orientation.

====================================================================
CRITICAL OBJECTIVE:  LINE TRANSFER (NO EXCEPTIONS)
====================================================================

You must transfer ONLY the measurement lines that already exist in the REFERENCE (SECOND) image.
You are NOT allowed to invent, infer, split, merge, extend, or add measurements.

The product in the input (FIRST) image MUST remain in its original orientation, size, and position.
Only the measurement lines are drawn, and they must adapt to match how the physical features 
appear in the input image's orientation. 

Drawing MORE lines than the reference is a CRITICAL FAILURE.
Drawing FEWER lines is allowed ONLY if a line cannot be confidently mapped.
NEVER compensate by adding new lines.

====================================================================
VISUAL OUTPUT REQUIREMENTS (HARD CONSTRAINTS)
====================================================================

BACKGROUND:
- Pure white only (RGB 255, 255, 255)

PRODUCT:
- You MUST TRACE or COPY the product from the FIRST (input) image exactly as it appears
- The product outline should be IDENTICAL to the input product in:
  * Orientation (same rotation angle)
  * Size (same dimensions)
  * Position (same location in the image)
  * Shape (same outline/profile)
- Drawn as a VERY FAINT, ghost-like outline only
- Color: light gray (#E0E0E0)
- No visible edge lines, no fill, no shading
- CRITICAL: The product must be a pixel-perfect or near-pixel-perfect match to the FIRST image's product
- DO NOT redraw the product - TRACE/COPY it from the input image
- DO NOT change the product's orientation to match the reference image

MEASUREMENT LINES:
- Color: Pure black (#000000)
- Style: Solid lines with dot endpoints
- Every black line MUST represent a real measurement from the reference

====================================================================
NEGATIVE CONSTRAINTS (ABSOLUTE)
====================================================================

- DO NOT draw extra lines
- DO NOT invent or guess measurements
- DO NOT draw decorative or helper lines
- DO NOT redraw the product with visible edges
- DO NOT copy pixels from the reference
- DO NOT reposition or resize the product. You can not change the position or size of the product. You can only draw lines on the product.
- 🚨 DO NOT rotate the product - THIS IS ABSOLUTELY FORBIDDEN 🚨
- 🚨 DO NOT change the product's orientation, angle, or rotation - CRITICAL FAILURE 🚨
- DO NOT use reference image angles for lines - lines must use input image angles
- DO NOT draw lines at reference angles when orientations differ - adapt to input angles
- DO trace/copy the product from the FIRST image exactly as it appears
- DO identify physical features and draw lines connecting them at their actual angles in the input image 

====================================================================
STEP 1 — REFERENCE LINE REGISTRY (MANDATORY)
====================================================================

1. Visually inspect the REFERENCE image.
2. Count the measurement lines.
3. Create an internal line registry using IDs.

You MUST internally define:

REFERENCE LINE REGISTRY:
L1 – <feature being measured>
L2 – <feature being measured>
L3 – <feature being measured>
...

Total reference lines = N

From this point forward, you are LOCKED to exactly N lines.
Only lines L1 through LN are permitted.

Any line without an ID is FORBIDDEN.

====================================================================
LINE LOCK RULE (CRITICAL)
====================================================================

- You may ONLY draw lines listed in the registry (L1…LN). One line for each L.
- You may NOT split one line into multiple lines
- You may NOT combine lines
- You may NOT replace missing lines with new ones
- You must label the correct set of lines in the generated image as in reference image. Do not put wrong labels.

If a line cannot be confidently mapped, SKIP it.
Skipping is allowed. Adding is NOT.

====================================================================
STEP 2 — FEATURE MAPPING (ADAPTATION REQUIRED)
====================================================================

For EACH registered line, you must perform FEATURE-BASED MATCHING:

1. **IDENTIFY THE PHYSICAL FEATURE** in the reference image:
   - What physical dimension/edge/feature does this line measure?
   - Examples: "width of top edge", "height of left side", "diagonal measurement", "distance between two corners"
   - Think in terms of 3D geometry, NOT pixel coordinates

2. **LOCATE THE SAME PHYSICAL FEATURE** in the input image:
   - Find the exact same physical feature (edge, dimension, measurement point)
   - The feature may appear at a DIFFERENT angle or orientation in the input image
   - This is NORMAL and EXPECTED when orientations differ

3. **MEASURE THE ACTUAL ANGLE** in the input image:
   - Determine how this physical feature appears in the input image's coordinate system
   - Calculate the actual angle/direction of the feature AS IT APPEARS in the input
   - DO NOT use the reference image's angle - use the input image's angle

4. **DRAW THE LINE AT THE INPUT IMAGE'S ANGLE**:
   - Connect the physical feature points using the angle from step 3
   - The line must follow the input image's coordinate system (0,0 at top-left, x→right, y→down)
   - The line angle must match how the feature actually appears in the input image

CRITICAL RULES:
- Do NOT copy pixel positions from reference
- Do NOT use reference image angles - use input image angles
- Do NOT change what physical feature the line measures
- DO adapt the line angle to match the input image's orientation
- DO use the input image's coordinate system (not reference's)

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

EXAMPLE - CORRECT:
- Reference: Product is horizontal, line measures width (0° angle)
- Input: Product is rotated 90° clockwise
- CORRECT: Line should measure width in input's orientation (90° angle relative to reference)
- The line connects the same physical feature (width) but at the angle it appears in input

EXAMPLE - INCORRECT:
- Reference: Product is horizontal, line measures width (0° angle)
- Input: Product is rotated 90° clockwise
- WRONG: Drawing line at 0° (reference angle) - this doesn't match the feature in input
- WRONG: Rotating the input product to match reference orientation

KEY PRINCIPLE:
Lines must connect the SAME PHYSICAL FEATURES, but at the ANGLES those features 
actually appear in the INPUT IMAGE's coordinate system.

====================================================================
STEP 3 — SCHEMATIC DRAWING
====================================================================

BEFORE DRAWING:
1. Look at the FIRST (input) image's product
2. Trace/copy that product EXACTLY as it appears (same orientation, size, position)
3. Draw it as a faint gray outline
4. Verify: Does your product match the input product's orientation? If NO, start over.

THEN DRAW LINES:
- Draw ONLY the registered measurement lines
- Color: pure black (#000000)
- Style: solid with dot endpoints
- Product outline must remain faint gray and minimal
- Lines must use the INPUT IMAGE's coordinate system and angles
- Lines must connect physical features at the angles they appear in the INPUT image

====================================================================
FINAL VALIDATION (MANDATORY)
====================================================================

Before finalizing, perform these checks:

1. **LINE COUNT VALIDATION**:
   Reference line count = N
   Drawn line count = M
   If M > N → DELETE extra lines
   If M < N → ACCEPT (do NOT add)
   If M == N → OK

2. **ORIENTATION VALIDATION (CRITICAL - FAILURE IF ANY CHECK FAILS)**:
   ✓ Does the output product match the FIRST (input) image product's orientation EXACTLY?
     → If NO, you have FAILED. The product must not be rotated.
   ✓ Is the product in the SAME position as in the input image?
   ✓ Is the product the SAME size as in the input image?
   ✓ Is the product at the SAME angle/rotation as in the input image?
   ✓ Are lines drawn using the INPUT IMAGE's coordinate system?
   ✓ Do lines connect the same physical features as in reference?
   ✓ Do line angles match how features appear in the INPUT image (not reference angles)?
   
   IF THE PRODUCT ORIENTATION DOES NOT MATCH THE INPUT IMAGE EXACTLY, YOU HAVE FAILED.

3. **FEATURE MATCHING VALIDATION**:
   ✓ Each line measures the same physical feature as its corresponding reference line
   ✓ Lines are not just copied at reference angles - they're adapted to input angles
   ✓ If input is rotated relative to reference, lines are rotated accordingly

4. **COORDINATE SYSTEM VALIDATION**:
   ✓ All coordinates use input image's system (top-left = 0,0; x→right; y→down)
   ✓ No rotation or transformation of the input image occurred
   ✓ Lines follow the actual geometry of features in the input image

====================================================================
FAIL-SAFE RULE
====================================================================

If you are uncertain about feature correspondence, perspective, or geometry:
DO NOT GUESS.
DO NOT ADD LINES.
Return fewer lines or an empty schematic.

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

