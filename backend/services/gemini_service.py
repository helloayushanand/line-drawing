from google import genai
from google.genai import types
from PIL import Image
import io
import json
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
        reference_lines: Optional[List[dict]] = None  # NEW: Labeled reference lines
    ) -> Optional[Image.Image]:
        """
        Use Gemini 3 to generate image with measurement lines drawn
        Returns None if image generation is not available
        """
        try:
            input_img = decode_base64_image(input_image_base64)
            ref_img = decode_base64_image(reference_image_base64)
            
            # Build label information from reference_lines
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
            
            # Build prompt
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

Your task is to add measurement guide lines to the PRODUCT in the FIRST image,
using the SECOND image strictly as a REFERENCE for which measurement lines exist.

{label_info}

====================================================================
CRITICAL OBJECTIVE: EXACT LINE TRANSFER (NO EXCEPTIONS)
====================================================================

You must transfer ONLY the measurement lines that already exist in the REFERENCE image.
You are NOT allowed to invent, infer, split, merge, extend, or add measurements.

Drawing MORE lines than the reference is a CRITICAL FAILURE.
Drawing FEWER lines is allowed ONLY if a line cannot be confidently mapped.
NEVER compensate by adding new lines.

====================================================================
VISUAL OUTPUT REQUIREMENTS (HARD CONSTRAINTS)
====================================================================

BACKGROUND:
- Pure white only (RGB 255, 255, 255)

PRODUCT:
- Drawn as a VERY FAINT, ghost-like outline only
- Color: light gray (#E0E0E0)
- No visible edge lines, no fill, no shading

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

For EACH registered line:

- Identify the physical feature it measures in the reference
- Locate the SAME physical feature on the input product
- Adapt for rotation, perspective, or orientation differences
- Attach the line to the correct 3D feature

Do NOT copy pixel positions.
Do NOT change what the line measures.

====================================================================
STEP 3 — SCHEMATIC DRAWING
====================================================================

- Draw ONLY the registered measurement lines
- Color: pure black (#000000)
- Style: solid with dot endpoints
- Product outline must remain faint gray and minimal

====================================================================
FINAL VALIDATION (MANDATORY)
====================================================================

Before finalizing, perform this check:

Reference line count = N
Drawn line count = M

If M > N → DELETE extra lines
If M < N → ACCEPT (do NOT add)
If M == N → OK

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
                
                # Debug: Check for prompt feedback and safety blocks
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    print(f"⚠️  Prompt feedback: {response.prompt_feedback}")
                    if hasattr(response.prompt_feedback, 'block_reason'):
                        print(f"❌ BLOCKED: {response.prompt_feedback.block_reason}")
                    if hasattr(response.prompt_feedback, 'safety_ratings'):
                        print(f"⚠️  Safety ratings: {response.prompt_feedback.safety_ratings}")
                
                # Parse response - extract image from candidates (matching working example)
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
                    # Log all candidate finish reasons for debugging
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
        """
        Calculate the aspect ratio of the image and return the closest supported standard aspect ratio.
        Supported: '1:1', '3:4', '4:3', '9:16', '16:9'
        """
        w, h = image.size
        ratio = w / h
        
        # Standard ratios and their float values
        standards = {
            '1:1': 1.0,
            '3:4': 3/4,
            '4:3': 4/3,
            '9:16': 9/16,
            '16:9': 16/9
        }
        
        # Find closest
        closest_ratio = min(standards.keys(), key=lambda k: abs(standards[k] - ratio))
        print(f"  📊 Input aspect ratio: {ratio:.2f} ({w}x{h}), closest standard: {closest_ratio}")
        return closest_ratio
    
    def predict_lines_direct(
        self,
        input_image_base64: str,
        reference_image_base64: str,
        expected_line_count: Optional[int] = None,
        line_types: Optional[List[str]] = None
    ) -> List[dict]:
        """
        Direct coordinate prediction via text output
        Enhanced prompt for better accuracy
        """
        input_img = decode_base64_image(input_image_base64)
        ref_img = decode_base64_image(reference_image_base64)
        
        line_count_info = f" The reference has {expected_line_count} measurement lines." if expected_line_count else ""
        line_types_info = f" They measure: {', '.join(line_types)}." if line_types else ""
        
        prompt = f"""You are a technical drawing expert. Analyze both images:

REFERENCE IMAGE (second image) Analysis:
- Count the measurement lines{line_count_info}{line_types_info}
- Note the positioning style (offset from edges, angles)
- Observe labeling convention

TASK: Apply the EXACT same measurement pattern to the NEW product (first image):
1. Draw the same number of lines
2. Match measurement types and labels
3. Use similar positioning relative to product edges
4. Maintain similar offset distances
5. Match angles/orientations

CRITICAL: Output precise coordinates. Lines must align with actual product edges.

Output JSON array format:
[
  {{"start": [y, x], "end": [y, x], "label": "string"}},
  ...
]
Coordinates normalized to 0-1000 range. Be extremely precise."""

        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[input_img, ref_img, prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT'],
                )
            )
            
            # Parse JSON response
            response_text = ""
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        response_text = part.text.strip()
                        break
                if response_text:
                    break
            
            # Handle markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            lines = json.loads(response_text)
            
            # Ensure it's a list
            if not isinstance(lines, list):
                lines = [lines]
            
            return lines
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response_text[:500]}")
            return []
        except Exception as e:
            print(f"Error in predict_lines_direct: {e}")
            import traceback
            traceback.print_exc()
            return []

