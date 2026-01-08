/**
 * Main component for line detection application
 */

import React, { useState, useCallback, useRef } from 'react';
import { ReferenceImageCanvas, DrawnLine, ReferenceImageCanvasHandle } from './ReferenceImageCanvas';
import { ResultVisualization } from './ResultVisualization';
import { detectLines, LineDetectionResponse } from './api';

export function LineDetectionApp() {
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [referenceImageUrl, setReferenceImageUrl] = useState<string | null>(null);
  const [inputImage, setInputImage] = useState<File | null>(null);
  const [inputImageUrl, setInputImageUrl] = useState<string | null>(null);
  const [drawnLines, setDrawnLines] = useState<DrawnLine[]>([]);
  const [lineCount, setLineCount] = useState<number>(3);
  const [category, setCategory] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('gemini');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<LineDetectionResponse | null>(null);
  const [inputImageBase64, setInputImageBase64] = useState<string | null>(null);
  const canvasRef = useRef<ReferenceImageCanvasHandle>(null);

  // Handle reference image upload
  const handleReferenceImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setReferenceImage(file);
      const url = URL.createObjectURL(file);
      setReferenceImageUrl(url);
      setDrawnLines([]); // Reset lines when image changes
    }
  };

  // Handle input image upload
  const handleInputImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setInputImage(file);
      const url = URL.createObjectURL(file);
      setInputImageUrl(url);
    }
  };

  // Handle drawn lines change
  const handleLinesChange = useCallback((lines: DrawnLine[]) => {
    setDrawnLines(lines);
    // Auto-update line count if lines are present
    if (lines.length > 0) {
      setLineCount(lines.length);
    }
  }, []);

  // Submit form
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResults(null);
    setInputImageBase64(null);

    if (!referenceImage || !inputImage) {
      setError('Please upload both reference and input images');
      return;
    }

    if (lineCount <= 0) {
      setError('Number of lines must be greater than 0');
      return;
    }

    setLoading(true);

    try {
      // Get canvas image with drawn lines, or fallback to original reference image
      let referenceImageToSend = referenceImage;

      if (canvasRef.current && drawnLines.length > 0) {
        // Export canvas with lines as a File
        const canvasFile = await canvasRef.current.getCanvasAsFile();
        if (canvasFile) {
          referenceImageToSend = canvasFile;
          console.log('✓ Using canvas image with drawn lines');
        } else {
          console.warn('⚠ Could not export canvas, using original reference image');
        }
      } else {
        console.log('ℹ No lines drawn or canvas not available, using original reference image');
      }

      const response = await detectLines(
        inputImage,
        referenceImageToSend,
        lineCount,
        category || undefined,
        selectedModel,
        drawnLines.length > 0 ? drawnLines : undefined  // NEW: Pass labeled lines
      );

      setResults(response);
      // Set input image base64 from API response (resized 1024x1024)
      if (response.input_image_base64) {
        setInputImageBase64(response.input_image_base64);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to detect lines');
      console.error('Error detecting lines:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-black">Line Detection Tool</h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Image Upload Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Reference Image */}
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-black">Reference Image</h2>
              <input
                type="file"
                accept="image/*"
                onChange={handleReferenceImageChange}
                className="mb-4 block w-full text-sm text-black file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {referenceImageUrl && (
                <div className="mt-4">
                  <p className="text-sm text-black mb-2">
                    Draw lines on the reference image (click and drag):
                  </p>
                  <ReferenceImageCanvas
                    ref={canvasRef}
                    imageUrl={referenceImageUrl}
                    onLinesChange={handleLinesChange}
                  />
                  {drawnLines.length > 0 && (
                    <p className="text-sm text-black mt-2">
                      {drawnLines.length} line(s) drawn
                    </p>
                  )}
                </div>
              )}
            </div>

            {/* Input Image */}
            <div className="bg-white p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold mb-4 text-black">Input Image</h2>
              <input
                type="file"
                accept="image/*"
                onChange={handleInputImageChange}
                className="mb-4 block w-full text-sm text-black file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {inputImageUrl && (
                <div className="mt-4">
                  <img
                    src={inputImageUrl}
                    alt="Input"
                    className="max-w-full h-auto border border-gray-300 rounded"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Form Inputs */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4 text-black">Detection Parameters</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="lineCount" className="block text-sm font-medium text-black mb-2">
                  Number of Lines *
                </label>
                <input
                  type="number"
                  id="lineCount"
                  min="1"
                  value={lineCount}
                  onChange={(e) => setLineCount(parseInt(e.target.value) || 1)}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black bg-white"
                />
              </div>
              <div>
                <label htmlFor="category" className="block text-sm font-medium text-black mb-2">
                  Category (Optional)
                </label>
                <input
                  type="text"
                  id="category"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  placeholder="e.g., chair, table"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black bg-white placeholder:text-gray-400"
                />
              </div>
            </div>
            <div className="mt-4">
              <label htmlFor="model" className="block text-sm font-medium text-black mb-2">
                Model
              </label>
              <select
                id="model"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-black bg-white"
              >
                <option value="gemini">Gemini</option>
                <option value="flux">Flux 2 Pro</option>
              </select>
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading || !referenceImage || !inputImage}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed font-semibold"
            >
              {loading ? 'Processing...' : 'Detect Lines'}
            </button>
          </div>
        </form>

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Results Display */}
        {results && inputImageUrl && (
          <div className="mt-6 bg-white p-4 rounded-lg shadow">
            <h2 className="text-2xl font-semibold mb-4 text-black">Results</h2>
            <div className="mb-4 text-sm text-black">
              <p>Processing time: {results.processing_time.toFixed(2)}s</p>
            </div>

            {/* JSON Coordinates Display */}
            <div className="mb-4 bg-gray-50 p-4 rounded-lg border border-gray-300">
              <h3 className="text-lg font-semibold mb-2 text-black">Line Coordinates (JSON)</h3>
              <pre className="text-xs text-black overflow-auto max-h-96 bg-white p-3 rounded border border-gray-200">
                {JSON.stringify(results.lines, null, 2)}
              </pre>
            </div>

            <ResultVisualization
              inputImageUrl={inputImageBase64 || inputImageUrl}
              referenceImageUrl={referenceImageUrl}
              lines={results.lines}
            />
          </div>
        )}
      </div>
    </div>
  );
}

