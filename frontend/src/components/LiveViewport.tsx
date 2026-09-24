import React, { useState, useRef, useEffect } from "react";
import { Globe, Lock, ExternalLink, RefreshCw, Eye, EyeOff } from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { DOMElement, Decision } from "@/types";

interface LiveViewportProps {
  screenshot?: string;
  url?: string;
  title?: string;
  elements?: DOMElement[];
  decision?: Decision | null;
  hoveredElementIndex?: string | null;
  setHoveredElementIndex?: (index: string | null) => void;
  onRefresh?: () => void;
  onLaunch?: () => void;
  isInitializing?: boolean;
}

export const LiveViewport: React.FC<LiveViewportProps> = ({
  screenshot,
  url,
  title,
  elements = [],
  decision,
  hoveredElementIndex,
  setHoveredElementIndex,
  onRefresh,
  onLaunch,
  isInitializing,
}) => {
  const [showOverlays, setShowOverlays] = useState(true);
  const [fitMode] = useState<"contain" | "fill">("contain");
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState({ scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 });

  // Update scale when window resizes or screenshot loads
  const updateScale = () => {
    if (!imgRef.current || !containerRef.current) return;
    const img = imgRef.current;
    const container = containerRef.current;

    const natW = img.naturalWidth || 1280;
    const natH = img.naturalHeight || 800;
    const contW = container.clientWidth;
    const contH = container.clientHeight;

    const ratio = Math.min(contW / natW, contH / natH);
    const renderW = natW * ratio;
    const renderH = natH * ratio;
    const offsetX = (contW - renderW) / 2;
    const offsetY = (contH - renderH) / 2;
    setScale({
      scaleX: renderW / natW,
      scaleY: renderH / natH,
      offsetX,
      offsetY,
    });
  };

  useEffect(() => {
    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, [screenshot, fitMode]);

  // Target element choice id
  const targetId = decision?.choice || decision?.target;

  return (
    <div className="flex flex-col h-full rounded-lg border border-border bg-card overflow-hidden">
      {/* Chrome Toolbar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/20">
        <div className="flex items-center gap-2">
          {/* Neutral window dots */}
          <div className="flex items-center gap-1.5 mr-2">
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
            <div className="w-2.5 h-2.5 rounded-full bg-zinc-700"></div>
          </div>

          {/* Address Bar */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-md bg-background border border-border text-xs text-muted-foreground w-64 md:w-96 lg:w-[420px] max-w-full">
            <Lock className="h-3 w-3 text-muted-foreground shrink-0" />
            <span className="truncate font-mono text-[11px] text-foreground select-all">
              {url || "about:blank"}
            </span>
          </div>

          {url && url !== "about:blank" && (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors p-1"
              title="Open in new browser tab"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
          )}
        </div>

        {/* Viewport Actions */}
        <div className="flex items-center gap-1.5">
          {elements.length > 0 && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant={showOverlays ? "secondary" : "ghost"}
                    size="sm"
                    onClick={() => setShowOverlays(!showOverlays)}
                    className="h-7 text-xs gap-1.5 font-mono px-2"
                  >
                    {showOverlays ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                    <span className="hidden sm:inline">Overlay</span>
                    <span className="px-1 py-0 text-[10px] h-4 rounded bg-secondary border border-border">
                      {elements.length}
                    </span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Toggle element overlays</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}

          {onRefresh && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={onRefresh}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Refresh Viewport</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>

      {/* Viewport Canvas */}
      <div
        ref={containerRef}
        className="relative flex-1 bg-black flex items-center justify-center overflow-hidden min-h-[360px]"
      >
        {screenshot ? (
          <div className="relative w-full h-full flex items-center justify-center select-none">
            <img
              ref={imgRef}
              src={`data:image/jpeg;base64,${screenshot}`}
              alt={title || "Browser View"}
              onLoad={updateScale}
              className="max-w-full max-h-full object-contain"
            />

            {/* Bounding Box SVG Overlay */}
            {showOverlays && elements.length > 0 && (
              <svg
                className="absolute pointer-events-none"
                style={{
                  left: `${scale.offsetX}px`,
                  top: `${scale.offsetY}px`,
                  width: `${(imgRef.current?.naturalWidth || 1280) * scale.scaleX}px`,
                  height: `${(imgRef.current?.naturalHeight || 800) * scale.scaleY}px`,
                }}
              >
                {elements.map((el) => {
                  if (!el.rect) return null;
                  const isHovered = hoveredElementIndex === el.index;
                  const isTarget = targetId === el.index || targetId === `e${el.index}`;

                  const x = el.rect.x * scale.scaleX;
                  const y = el.rect.y * scale.scaleY;
                  const w = el.rect.w * scale.scaleX;
                  const h = el.rect.h * scale.scaleY;

                  if (w <= 0 || h <= 0) return null;

                  return (
                    <g
                      key={el.index}
                      className="pointer-events-auto cursor-pointer"
                      onMouseEnter={() => setHoveredElementIndex?.(el.index)}
                      onMouseLeave={() => setHoveredElementIndex?.(null)}
                    >
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill={
                          isTarget
                            ? "rgba(255, 255, 255, 0.16)"
                            : isHovered
                            ? "rgba(255, 255, 255, 0.08)"
                            : "transparent"
                        }
                        stroke={isTarget || isHovered ? "#ffffff" : "rgba(255, 255, 255, 0.35)"}
                        strokeWidth={isTarget ? 2 : 1}
                        strokeDasharray={isTarget || isHovered ? "none" : "3,2"}
                        rx={2}
                      />
                      {(isHovered || isTarget || elements.length < 25) && (
                        <g>
                          <rect
                            x={x}
                            y={Math.max(0, y - 14)}
                            width={Math.max(18, el.index.length * 7 + 6)}
                            height={13}
                            fill={isTarget ? "#ffffff" : isHovered ? "#d4d4d8" : "#27272a"}
                            rx={2}
                          />
                          <text
                            x={x + 3}
                            y={Math.max(0, y - 14) + 9}
                            fill={isTarget || isHovered ? "#000000" : "#ffffff"}
                            fontSize="9"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            {el.index}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          /* Clean Monochrome Empty State */
          <div className="flex flex-col items-center justify-center p-8 text-center max-w-sm">
            <img
              src="/katai.png"
              alt="Katai"
              className="h-12 w-12 rounded-lg border border-border bg-secondary/30 object-contain p-1.5 mb-3"
            />
            <h3 className="text-sm font-semibold text-foreground mb-1">
              Browser Inactive
            </h3>
            <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
              Initialize a session to observe DOM state and render interactive overlays.
            </p>
            {onLaunch && (
              <Button
                variant="default"
                size="sm"
                onClick={onLaunch}
                disabled={isInitializing}
                className="text-xs font-medium"
              >
                {isInitializing ? "Initializing..." : "Initialize Session"}
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Footer Info Strip */}
      <div className="px-3 py-1.5 border-t border-border bg-muted/10 flex items-center justify-between text-[11px] text-muted-foreground font-mono">
        <span className="truncate max-w-[60%]">
          {title ? `Title: ${title}` : "Ready for observation"}
        </span>
        <div className="flex items-center gap-3">
          <span>Controls: {elements.length}</span>
          <span className="hidden md:inline">CDP: Connected</span>
        </div>
      </div>
    </div>
  );
};
