import React, { useState } from "react";
import { Code2, Copy, Check } from "lucide-react";
import { Button } from "./ui/button";

interface RawStateViewerProps {
  state: any;
}

export const RawStateViewer: React.FC<RawStateViewerProps> = ({ state }) => {
  const [copied, setCopied] = useState(false);

  const sanitizedState = { ...state };
  if (sanitizedState.screenshot) {
    sanitizedState.screenshot = `[base64 image - ${(sanitizedState.screenshot.length / 1024).toFixed(1)} KB]`;
  }

  const jsonString = JSON.stringify(sanitizedState, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(state, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-full bg-card rounded-lg border border-border overflow-hidden">
      <div className="flex items-center justify-between p-3 border-b border-border bg-muted/10">
        <div className="flex items-center gap-2">
          <Code2 className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            State JSON
          </span>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopy}
          className="h-7 text-xs gap-1.5"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          <span>{copied ? "Copied" : "Copy JSON"}</span>
        </Button>
      </div>

      <div className="flex-1 p-3 overflow-auto bg-black font-mono text-xs text-muted-foreground">
        <pre className="text-foreground/90 selection:bg-secondary">{jsonString}</pre>
      </div>
    </div>
  );
};
