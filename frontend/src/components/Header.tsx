import React from "react";
import { Terminal, Cpu, RotateCcw, Keyboard, CheckCircle2, AlertTriangle, Play } from "lucide-react";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { AgentState } from "@/types";

interface HeaderProps {
  state: AgentState;
  onReset: () => void;
  onOpenShortcuts: () => void;
  isBusy: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  state,
  onReset,
  onOpenShortcuts,
  isBusy,
}) => {
  const getStatusBadge = () => {
    switch (state.status) {
      case "running":
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse"></span>
            Running
          </Badge>
        );
      case "predicted":
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-sky-400"></span>
            Predicted
          </Badge>
        );
      case "done":
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
            Completed
          </Badge>
        );
      case "blocked":
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-rose-400"></span>
            Blocked
          </Badge>
        );
      case "ready":
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400"></span>
            Ready
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="gap-1.5 px-2.5 py-0.5 border-border bg-secondary/40 font-mono uppercase text-[11px] font-medium text-muted-foreground">
            <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60"></span>
            Idle
          </Badge>
        );
    }
  };

  const modelName = state.model || "v10s · 322M";
  const deviceName = state.device || (navigator.platform.includes("Mac") ? "MPS" : "CUDA");

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background px-4 lg:px-6 py-2.5 flex items-center justify-between">
      {/* Brand & Architecture */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <img
            src="/katai.png"
            alt="Katai"
            className="h-7 w-7 rounded-md border border-border object-contain bg-secondary/50 p-0.5"
          />
          <span className="text-sm font-semibold tracking-tight text-foreground font-mono">
            katai
          </span>
          <span className="text-[11px] text-muted-foreground font-mono px-1.5 py-0.5 rounded border border-border bg-secondary/50">
            v0.1
          </span>
        </div>

        <div className="hidden md:flex items-center gap-1.5 pl-3 border-l border-border font-mono text-xs text-muted-foreground">
          <span className="px-2 py-0.5 rounded border border-border bg-secondary/30">
            {modelName}
          </span>
          <span className="px-2 py-0.5 rounded border border-border bg-secondary/30">
            {deviceName}
          </span>
        </div>
      </div>

      {/* Status & Actions */}
      <div className="flex items-center gap-3">
        {/* Status Pill */}
        {getStatusBadge()}

        {/* Telemetry */}
        <div className="hidden sm:flex items-center gap-3 font-mono text-xs text-muted-foreground border-l border-border pl-3">
          <div>
            Steps: <span className="text-foreground font-medium">{state.step_count}</span>
          </div>
          <div>
            Latency: <span className="text-foreground font-medium">{state.elapsed_ms}ms</span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-1.5 border-l border-border pl-3">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={onOpenShortcuts}
                  className="text-muted-foreground hover:text-foreground"
                >
                  <Keyboard className="h-3.5 w-3.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Keyboard shortcuts (?)</TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
            disabled={isBusy}
            className="h-8 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <RotateCcw className="h-3 w-3" />
            <span className="hidden sm:inline">Reset</span>
          </Button>
        </div>
      </div>
    </header>
  );
};
