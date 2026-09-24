import React from "react";
import { Zap, MousePointer, Type, ListFilter, CheckCircle2, AlertOctagon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Decision } from "@/types";

interface DecisionMetricsProps {
  decision?: Decision | null;
  lastAction?: any;
}

export const DecisionMetrics: React.FC<DecisionMetricsProps> = ({
  decision,
  lastAction,
}) => {
  if (!decision && !lastAction) {
    return (
      <Card className="border border-border bg-card">
        <CardHeader className="py-3 px-4 border-b border-border">
          <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-2">
            <Zap className="h-3.5 w-3.5 text-muted-foreground" />
            Decision Head
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 text-center text-xs text-muted-foreground">
          No decision evaluated yet.
        </CardContent>
      </Card>
    );
  }

  const op = decision?.operation || lastAction?.operation || "UNKNOWN";
  const conf = decision?.confidence ?? lastAction?.confidence ?? 0;
  const confPct = Math.round(conf * 100);
  const latency = decision?.latency_ms ?? lastAction?.latency_ms ?? 0;
  const target = decision?.target || decision?.choice || lastAction?.choice || "None";
  const textTyped = lastAction?.text;

  // Colored badges for operations as explicitly requested ("no shiny cool colors except on badges 'click' 'type' etc")
  const getOpBadge = (operation: string) => {
    switch (operation) {
      case "CLICK":
        return (
          <Badge variant="info" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            <MousePointer className="h-3 w-3" /> CLICK
          </Badge>
        );
      case "TYPE_TEXT":
        return (
          <Badge variant="success" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            <Type className="h-3 w-3" /> TYPE_TEXT
          </Badge>
        );
      case "SELECT":
        return (
          <Badge variant="warning" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            <ListFilter className="h-3 w-3" /> SELECT
          </Badge>
        );
      case "DONE":
        return (
          <Badge variant="purple" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            <CheckCircle2 className="h-3 w-3" /> DONE
          </Badge>
        );
      case "BLOCKED":
        return (
          <Badge variant="destructive" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            <AlertOctagon className="h-3 w-3" /> BLOCKED
          </Badge>
        );
      default:
        return (
          <Badge variant="secondary" className="gap-1.5 px-2 py-0.5 text-xs font-mono font-semibold">
            {operation}
          </Badge>
        );
    }
  };

  return (
    <Card className="border border-border bg-card">
      <CardHeader className="py-3 px-4 border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-2">
            <Zap className="h-3.5 w-3.5 text-muted-foreground" />
            Decision Head
          </CardTitle>
          <span className="text-[11px] font-mono text-muted-foreground">
            {latency} ms
          </span>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-3">
        {/* Operation & Target */}
        <div className="flex items-center justify-between gap-2 p-2.5 rounded-md bg-secondary/30 border border-border">
          <div className="flex items-center gap-2">
            {getOpBadge(op)}
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-medium text-muted-foreground">Target</span>
              <span className="text-xs font-mono font-medium text-foreground truncate max-w-[170px]">
                {target}
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="text-[10px] uppercase font-medium text-muted-foreground block">Score</span>
            <span className="text-xs font-mono font-medium text-foreground">
              {confPct}%
            </span>
          </div>
        </div>

        {/* Confidence Gauge */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs font-mono text-muted-foreground">
            <span>Calibrated Confidence</span>
            <span className="text-foreground">{(conf).toFixed(4)}</span>
          </div>
          <Progress
            value={confPct}
            className="h-1.5 bg-secondary"
            indicatorClassName="bg-primary"
          />
        </div>

        {/* Typed Text Preview */}
        {textTyped && (
          <div className="p-2.5 rounded-md bg-secondary/30 border border-border text-xs space-y-1">
            <div className="flex items-center gap-1.5 text-muted-foreground text-[11px] font-medium">
              <Type className="h-3 w-3" /> Form Text Value:
            </div>
            <div className="font-mono text-foreground text-xs bg-background p-1.5 rounded border border-border truncate select-all">
              "{textTyped}"
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
