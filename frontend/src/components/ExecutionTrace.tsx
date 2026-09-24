import React from "react";
import { History, CornerDownRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { HistoryItem } from "@/types";

interface ExecutionTraceProps {
  history: HistoryItem[];
  onSelectStep?: (step: HistoryItem) => void;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({
  history = [],
  onSelectStep,
}) => {
  const reversedHistory = [...history].reverse();

  // Colored badges for operations as explicitly requested ("no shiny cool colors except on badges 'click' 'type' etc")
  const getActionBadge = (kind?: string, operation?: string) => {
    const key = (kind || operation || "action").toLowerCase();
    switch (key) {
      case "click":
        return <Badge variant="info" className="text-[10px] px-1.5 py-0 font-mono font-semibold">CLICK</Badge>;
      case "fill":
      case "type_text":
        return <Badge variant="success" className="text-[10px] px-1.5 py-0 font-mono font-semibold">TYPE</Badge>;
      case "select":
        return <Badge variant="warning" className="text-[10px] px-1.5 py-0 font-mono font-semibold">SELECT</Badge>;
      case "done":
        return <Badge variant="purple" className="text-[10px] px-1.5 py-0 font-mono font-semibold">DONE</Badge>;
      case "blocked":
        return <Badge variant="destructive" className="text-[10px] px-1.5 py-0 font-mono font-semibold">BLOCKED</Badge>;
      default:
        return <Badge variant="secondary" className="text-[10px] px-1.5 py-0 font-mono">{key.toUpperCase()}</Badge>;
    }
  };

  return (
    <Card className="flex flex-col h-full border border-border bg-card overflow-hidden">
      <CardHeader className="py-3 px-4 border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-2">
            <History className="h-3.5 w-3.5 text-muted-foreground" />
            Execution Trace
          </CardTitle>
          <span className="text-[11px] font-mono text-muted-foreground">
            {history.length} {history.length === 1 ? "step" : "steps"}
          </span>
        </div>
      </CardHeader>

      <CardContent className="p-3 flex-1 overflow-y-auto space-y-2">
        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground">
            <History className="h-6 w-6 mb-2 opacity-30" />
            <p className="text-xs">No execution steps yet</p>
          </div>
        ) : (
          reversedHistory.map((item) => (
            <div
              key={item.step}
              onClick={() => onSelectStep?.(item)}
              className="p-2.5 rounded-md bg-secondary/30 hover:bg-secondary/60 border border-border transition-colors cursor-pointer space-y-1.5"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-foreground">Step {item.step}</span>
                  {getActionBadge(item.kind, item.operation)}
                </div>
                <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                  {item.confidence !== undefined && (
                    <span className="text-foreground">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  )}
                  <span>·</span>
                  <span>{item.latency_ms || 0}ms</span>
                </div>
              </div>

              {/* Action Description */}
              <div className="text-xs text-foreground/90 line-clamp-2">
                {item.action || "Executed browser action"}
              </div>

              {/* Typed text callout */}
              {item.text && (
                <div className="flex items-center gap-1.5 text-[11px] font-mono text-foreground bg-background px-2 py-0.5 rounded border border-border">
                  <CornerDownRight className="h-3 w-3 shrink-0 text-muted-foreground" />
                  <span className="truncate">typed: "{item.text}"</span>
                </div>
              )}

              {/* Status / Page Change Badge */}
              <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono pt-0.5">
                <span className="truncate max-w-[170px]">{item.choice || ""}</span>
                {item.page_changed !== undefined && (
                  <span className={item.page_changed ? "text-foreground font-medium" : "text-muted-foreground"}>
                    {item.page_changed ? "DOM Changed" : "Unchanged"}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
};
