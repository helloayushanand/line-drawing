/**
 * Canvas component for drawing lines on reference image
 */

import { useCallback, useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';

export interface DrawnLine {
  start: { x: number; y: number };
  end: { x: number; y: number };
  label: string;  // Required, will be auto-generated
  id?: string;    // Optional unique identifier
}

interface ReferenceImageCanvasProps {
  imageUrl: string | null;
  onLinesChange?: (lines: DrawnLine[]) => void;
  width?: number;
  height?: number;
}

export interface ReferenceImageCanvasHandle {
  getCanvasAsFile: () => Promise<File | null>;
  getCanvasAsDataUrl: () => string | null;
}

export const ReferenceImageCanvas = forwardRef<ReferenceImageCanvasHandle, ReferenceImageCanvasProps>(({
  imageUrl,
  onLinesChange,
  width = 600,
  height = 600,
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [lines, setLines] = useState<DrawnLine[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentLine, setCurrentLine] = useState<{ start: { x: number; y: number }; end: { x: number; y: number } } | null>(null);

  const drawLine = (ctx: CanvasRenderingContext2D, line: DrawnLine, index: number) => {
    const blackColor = '#000000';

    ctx.save();
    ctx.strokeStyle = blackColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(line.start.x, line.start.y);
    ctx.lineTo(line.end.x, line.end.y);
    ctx.stroke();

    // Draw endpoints
    ctx.fillStyle = blackColor;
    ctx.beginPath();
    ctx.arc(line.start.x, line.start.y, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(line.end.x, line.end.y, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Draw label if available
    if (line.label) {
      ctx.fillStyle = blackColor;
      ctx.font = '14px Arial';
      ctx.fillText(line.label, line.start.x + 10, line.start.y - 10);
    }
    ctx.restore();
  };

  // Redraw lines when they change
  const redrawLines = useCallback(() => {
    if (!canvasRef.current || !imageRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Redraw image
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    // Draw all lines
    lines.forEach((line, index) => {
      drawLine(ctx, line, index);
    });

    // Draw current line being drawn (preview)
    if (currentLine && isDrawing) {
      ctx.save();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(currentLine.start.x, currentLine.start.y);
      ctx.lineTo(currentLine.end.x, currentLine.end.y);
      ctx.stroke();
      ctx.restore();
    }
  }, [lines, currentLine, isDrawing]);

  // Load image and draw
  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      // Scale canvas to fit image while maintaining aspect ratio
      const scale = Math.min(width / img.width, height / img.height);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      redrawLines();
    };
    img.src = imageUrl;
  }, [imageUrl, width, height, redrawLines]);

  useEffect(() => {
    redrawLines();
  }, [redrawLines]);

  // Notify parent of line changes
  useEffect(() => {
    onLinesChange?.(lines);
  }, [lines, onLinesChange]);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!imageUrl) return;
    const coords = getCanvasCoordinates(e);
    if (!coords) return;

    setIsDrawing(true);
    setCurrentLine({ start: coords, end: coords });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !currentLine) return;
    const coords = getCanvasCoordinates(e);
    if (!coords) return;
    setCurrentLine({ ...currentLine, end: coords });
    redrawLines();
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !currentLine) return;

    const coords = getCanvasCoordinates(e);
    if (!coords) {
      setIsDrawing(false);
      setCurrentLine(null);
      return;
    }

    // Add completed line with auto-generated label
    const lineNumber = lines.length + 1;
    const newLine: DrawnLine = {
      start: currentLine.start,
      end: coords,
      label: `L${lineNumber}`,  // Changed to Short Label: L1, L2, etc.
      id: `line_${Date.now()}_${lineNumber}`,  // Unique ID
    };

    setLines([...lines, newLine]);
    setIsDrawing(false);
    setCurrentLine(null);
    redrawLines();
  };

  const clearLines = () => {
    setLines([]);
  };

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    getCanvasAsFile: async (): Promise<File | null> => {
      if (!canvasRef.current) return null;

      return new Promise((resolve) => {
        canvasRef.current!.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'reference-with-lines.png', { type: 'image/png' });
            resolve(file);
          } else {
            resolve(null);
          }
        }, 'image/png');
      });
    },
    getCanvasAsDataUrl: (): string | null => {
      if (!canvasRef.current) return null;
      return canvasRef.current.toDataURL('image/png');
    },
  }));

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        className="border border-gray-300 cursor-crosshair"
        style={{ maxWidth: '100%', height: 'auto' }}
      />
      {lines.length > 0 && (
        <button
          onClick={clearLines}
          className="mt-2 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear Lines ({lines.length})
        </button>
      )}
    </div>
  );
});

ReferenceImageCanvas.displayName = 'ReferenceImageCanvas';

