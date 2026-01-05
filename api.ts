/**
 * API integration for line detection backend
 */

export interface Point {
  x: number;
  y: number;
}

export interface LineDrawing {
  start: Point;
  end: Point;
  label: string;
  confidence?: number | null;
}

export interface ReferenceLine {
  start: Point;
  end: Point;
  label: string;
  id?: string;
}

export interface LineDetectionRequest {
  input_image_base64: string;
  reference_image_base64: string;
  product_type?: string | null;
  expected_line_count?: number | null;
  line_types?: string[] | null;
  reference_lines?: ReferenceLine[] | null;  // NEW
  model?: string | null; // "gemini" or "flux"
}

export interface LineDetectionResponse {
  lines: LineDrawing[];
  method_used: string;
  processing_time: number;
  confidence_score?: number | null;
  generated_image_base64?: string | null;
  input_image_base64?: string | null;
}

// API base URL - can be configured via environment variable
// Default to production URL
const getApiBaseUrl = () => {
  if (typeof window !== 'undefined') {
    // Check if VITE_API_URL is set in window (can be set via vite config)
    return (window as any).__API_URL__ || 'https://ayush.sigaba.in';
  }
  return 'https://ayush.sigaba.in';
};

const API_BASE_URL = getApiBaseUrl();

/**
 * Convert image file to base64 string
 */
export async function imageToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove data URL prefix if present
      const base64 = result.includes(',') ? result.split(',')[1] : result;
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * Call backend API to detect lines
 */
export async function detectLines(
  inputImage: File,
  referenceImage: File,
  expectedLineCount: number,
  productType?: string,
  model: string = 'gemini',
  referenceLines?: Array<{ start: { x: number, y: number }, end: { x: number, y: number }, label: string, id?: string }>  // NEW
): Promise<LineDetectionResponse> {
  try {
    // Convert images to base64
    const inputBase64 = await imageToBase64(inputImage);
    const referenceBase64 = await imageToBase64(referenceImage);

    // Prepare request body
    const requestBody: any = {
      input_image_base64: `data:image/png;base64,${inputBase64}`,
      reference_image_base64: `data:image/png;base64,${referenceBase64}`,
      expected_line_count: expectedLineCount,
      model: model,
    };

    if (productType) {
      requestBody.product_type = productType;
    }

    // NEW: Add reference_lines if provided
    if (referenceLines && referenceLines.length > 0) {
      requestBody.reference_lines = referenceLines;
      console.log('📤 Sending reference lines with labels:', referenceLines);
    }

    const response = await fetch(`${API_BASE_URL}/api/detect-lines`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.error('Error in detectLines:', error);
    throw error;
  }
}

