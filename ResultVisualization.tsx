/**
 * Component to visualize detection results
 * Shows: Gemini generated image and extracted lines visualization
 */

import { useEffect, useRef, useState } from 'react';
import { LineDrawing } from './api';

interface ResultVisualizationProps {
  inputImageUrl: string | null;
  referenceImageUrl: string | null;
  lines: LineDrawing[];
}

export function ResultVisualization({
  inputImageUrl,
  referenceImageUrl,
  lines,
}: ResultVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Use input image for line visualization (clean image without black lines from Gemini)
  // Both images are 1024x1024, so normalized coordinates (0-1) align perfectly
  const imageUrlForLines = inputImageUrl;

  // Draw lines on image (prefer generated image if available)
  useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:26',message:'useEffect triggered',data:{linesCount:lines.length,hasImageUrl:!!imageUrlForLines,hasCanvas:!!canvasRef.current},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
    // #endregion
    if (!imageUrlForLines || !canvasRef.current || lines.length === 0) return;
    
    // If no reference image URL, fall back to original behavior
    if (!referenceImageUrl) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Draw lines
        const blackColor = 'rgb(0, 0, 0)';
        lines.forEach((line) => {
          const startX = line.start.x * canvas.width;
          const startY = line.start.y * canvas.height;
          const endX = line.end.x * canvas.width;
          const endY = line.end.y * canvas.height;
          
          ctx.save();
          ctx.strokeStyle = blackColor;
          ctx.lineWidth = 5;
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.stroke();
          
          ctx.fillStyle = blackColor;
          ctx.beginPath();
          ctx.arc(startX, startY, 8, 0, 2 * Math.PI);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(endX, endY, 8, 0, 2 * Math.PI);
          ctx.fill();
          
          const label = line.label || '';
          if (label) {
            ctx.fillStyle = blackColor;
            ctx.font = 'bold 16px Arial';
            ctx.fillText(label, startX + 10, startY - 10);
          }
          ctx.restore();
        });
        
        setImageLoaded(true);
      };
      img.src = imageUrlForLines;
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Load both input and reference images
    const inputImg = new Image();
    const refImg = new Image();
    
    let inputLoaded = false;
    let refLoaded = false;
    
    const drawExpandedCanvas = () => {
      if (!inputLoaded || !refLoaded) return;
      
      // Calculate reference aspect ratio
      const refAspect = refImg.width / refImg.height;
      const inputAspect = inputImg.width / inputImg.height;
      
      // Calculate expanded canvas dimensions that match reference aspect ratio
      let canvasWidth: number;
      let canvasHeight: number;
      
      if (inputAspect > refAspect) {
        // Input is wider relative to reference
        canvasWidth = inputImg.width;
        canvasHeight = Math.ceil(inputImg.width / refAspect);
        if (canvasHeight < inputImg.height) {
          canvasHeight = inputImg.height;
          canvasWidth = Math.ceil(inputImg.height * refAspect);
        }
      } else {
        // Input is taller relative to reference
        canvasHeight = inputImg.height;
        canvasWidth = Math.ceil(inputImg.height * refAspect);
        if (canvasWidth < inputImg.width) {
          canvasWidth = inputImg.width;
          canvasHeight = Math.ceil(inputImg.width / refAspect);
        }
      }
      
      // Set canvas size
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      
      // Fill with white background
      ctx.fillStyle = 'rgb(255, 255, 255)';
      ctx.fillRect(0, 0, canvasWidth, canvasHeight);
      
      // Calculate offset to center input image
      const xOffset = (canvasWidth - inputImg.width) / 2;
      const yOffset = (canvasHeight - inputImg.height) / 2;
      
      // Draw input image centered on expanded canvas
      ctx.drawImage(inputImg, xOffset, yOffset);

      // Draw lines in black
      const blackColor = 'rgb(0, 0, 0)';

      // Check for near-duplicate lines (very close coordinates)
      const lineDistances: number[] = [];
      lines.forEach((line, index) => {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:45',message:'Processing line',data:{lineIndex:index,startX:line.start.x,startY:line.start.y,endX:line.end.x,endY:line.end.y,label:line.label},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
        // #endregion
        
        // Check distance to previous lines
        if (index > 0) {
          const prevLine = lines[index - 1];
          const distX = Math.abs(line.start.x - prevLine.start.x) * canvas.width;
          const distY = Math.abs(line.start.y - prevLine.start.y) * canvas.height;
          const distance = Math.sqrt(distX * distX + distY * distY);
          lineDistances.push(distance);
          // #region agent log
          fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:55',message:'Line distance check',data:{lineIndex:index,prevLineIndex:index-1,distancePixels:distance,isVeryClose:distance<5},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
          // #endregion
        }
        
        // Convert normalized coordinates to pixel coordinates (normalized to expanded canvas)
        const startX = line.start.x * canvas.width;
        const startY = line.start.y * canvas.height;
        const endX = line.end.x * canvas.width;
        const endY = line.end.y * canvas.height;
        
        // #region agent log
        const isOutsideBounds = startX < 0 || startX > canvas.width || startY < 0 || startY > canvas.height ||
                                endX < 0 || endX > canvas.width || endY < 0 || endY > canvas.height;
        fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:52',message:'Pixel coordinates calculated',data:{lineIndex:index,startX:startX,startY:startY,endX:endX,endY:endY,canvasWidth:canvas.width,canvasHeight:canvas.height,isOutsideBounds:isOutsideBounds},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
        // #endregion

        // Draw line
        ctx.save();
        ctx.strokeStyle = blackColor;
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:60',message:'Line drawn',data:{lineIndex:index,startX:startX,startY:startY,endX:endX,endY:endY},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
        // #endregion

        // Draw endpoints
        ctx.fillStyle = blackColor;
        ctx.beginPath();
        ctx.arc(startX, startY, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(endX, endY, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Draw label
        const label = line.label || `Line ${index + 1}`;
        if (line.confidence !== null && line.confidence !== undefined) {
          ctx.fillStyle = blackColor;
          ctx.font = 'bold 16px Arial';
          ctx.fillText(`${label} (${(line.confidence * 100).toFixed(1)}%)`, startX + 10, startY - 10);
        } else {
          ctx.fillStyle = blackColor;
          ctx.font = 'bold 16px Arial';
          ctx.fillText(label, startX + 10, startY - 10);
        }
        ctx.restore();
      });
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/3cd10946-e982-4b9e-8550-52d78fdb1a8d',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ResultVisualization.tsx:82',message:'All lines drawn',data:{totalLines:lines.length,canvasWidth:canvas.width,canvasHeight:canvas.height},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion

      setImageLoaded(true);
    };
    
    inputImg.onload = () => {
      inputLoaded = true;
      drawExpandedCanvas();
    };
    
    refImg.onload = () => {
      refLoaded = true;
      drawExpandedCanvas();
    };
    
    inputImg.src = imageUrlForLines;
    refImg.src = referenceImageUrl;
  }, [imageUrlForLines, referenceImageUrl, lines]);

  return (
    <div className="border border-gray-300 rounded p-4 bg-white">
      <h3 className="text-lg font-semibold mb-2 text-black">Extracted Lines ({lines.length})</h3>
      {imageUrlForLines ? (
        <div className="relative">
          <canvas
            ref={canvasRef}
            className="max-w-full h-auto border border-gray-200 rounded"
            style={{ display: imageLoaded ? 'block' : 'none' }}
          />
          {!imageLoaded && (
            <div className="flex items-center justify-center h-64 bg-white rounded border border-gray-200">
              <p className="text-black">Loading...</p>
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 bg-white rounded border border-gray-200">
          <p className="text-black">No input image</p>
        </div>
      )}
    </div>
  );
}

