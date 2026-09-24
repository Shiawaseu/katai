import React, { useState } from "react";
import { Play, Pause, StepForward, Square, Globe, Target, Sliders, ChevronDown } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Textarea } from "./ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";

interface Preset {
  name: string;
  url: string;
  goal: string;
}

const PRESETS: Preset[] = [
  {
    name: "Wikipedia",
    url: "https://en.wikipedia.org/wiki/Main_Page",
    goal: "Search Wikipedia for 'Python programming language' and open the article about the Python language.",
  },
  {
    name: "Hacker News",
    url: "https://news.ycombinator.com",
    goal: "Find the top story with the most points on the front page and click on its discussion comments.",
  },
  {
    name: "GitHub",
    url: "https://github.com",
    goal: "Explore GitHub search for 'laya browser agent' and look at the first repository.",
  },
  {
    name: "TodoMVC",
    url: "https://todomvc.com/examples/react/dist/",
    goal: "Add two todo tasks: 'Test Katai browser agent' and 'Inspect DOM action space', then mark the first one as completed.",
  },
];

interface TaskConfigProps {
  url: string;
  setUrl: (url: string) => void;
  goal: string;
  setGoal: (goal: string) => void;
  status: string;
  isRunning: boolean;
  onInitialize: () => void;
  onStep: () => void;
  onToggleRun: () => void;
  onStop: () => void;
  stepDelay: number;
  setStepDelay: (delay: number) => void;
  isInitializing: boolean;
}

export const TaskConfig: React.FC<TaskConfigProps> = ({
  url,
  setUrl,
  goal,
  setGoal,
  status,
  isRunning,
  onInitialize,
  onStep,
  onToggleRun,
  onStop,
  stepDelay,
  setStepDelay,
  isInitializing,
}) => {
  const [showSettings, setShowSettings] = useState(false);

  const applyPreset = (preset: Preset) => {
    setUrl(preset.url);
    setGoal(preset.goal);
  };

  const isReadyToStep = status === "ready" || status === "predicted";

  return (
    <Card className="flex flex-col border border-border bg-card">
      <CardHeader className="py-3 px-4 border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-2">
            <Target className="h-3.5 w-3.5 text-muted-foreground" />
            Task Configuration
          </CardTitle>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setShowSettings(!showSettings)}
            className={showSettings ? "text-foreground bg-secondary" : "text-muted-foreground"}
            title="Execution Settings"
          >
            <Sliders className="h-3.5 w-3.5" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-3.5">
        {/* Presets */}
        <div className="space-y-1.5">
          <div className="text-[11px] font-medium text-muted-foreground">
            Presets
          </div>
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map((preset) => {
              const active = url === preset.url;
              return (
                <button
                  key={preset.name}
                  onClick={() => applyPreset(preset)}
                  type="button"
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors border ${
                    active
                      ? "bg-primary text-primary-foreground border-primary"
                      : "bg-transparent text-muted-foreground hover:text-foreground border-border hover:bg-secondary"
                  }`}
                >
                  {preset.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Target URL */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Globe className="h-3 w-3" /> Target URL
          </label>
          <Input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://..."
            className="font-mono text-xs h-9 bg-background border-border"
            disabled={isRunning || isInitializing}
          />
        </div>

        {/* Goal Instructions */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
            Goal Instructions
          </label>
          <Textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Describe what the agent should accomplish..."
            rows={3}
            className="text-xs bg-background border-border resize-none leading-relaxed font-sans"
            disabled={isRunning || isInitializing}
          />
        </div>

        {/* Settings Drawer */}
        {showSettings && (
          <div className="p-3 rounded-md bg-secondary/30 border border-border space-y-2 text-xs">
            <div className="flex items-center justify-between text-muted-foreground">
              <span>Auto-Run Step Delay:</span>
              <span className="font-mono text-foreground font-medium">{stepDelay} ms</span>
            </div>
            <input
              type="range"
              min="100"
              max="1500"
              step="50"
              value={stepDelay}
              onChange={(e) => setStepDelay(Number(e.target.value))}
              className="w-full accent-foreground h-1 bg-secondary rounded cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground font-mono">
              <span>100ms (Fast)</span>
              <span>300ms (Default)</span>
              <span>1500ms (Slow)</span>
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="pt-1 space-y-2">
          {/* Main Primary / Step Row */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant="default"
              size="default"
              onClick={onInitialize}
              disabled={isRunning || isInitializing || !url.trim()}
              className="w-full text-xs font-medium"
            >
              {isInitializing ? "Initializing..." : "Initialize"}
            </Button>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="secondary"
                    size="default"
                    onClick={onStep}
                    disabled={isRunning || isInitializing || !isReadyToStep}
                    className="w-full gap-1.5 text-xs font-medium border border-border"
                  >
                    <StepForward className="h-3.5 w-3.5" />
                    Step
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Execute 1 step (Space)</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {/* Auto Run and Stop Row */}
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant={isRunning ? "destructive" : "outline"}
              size="default"
              onClick={onToggleRun}
              disabled={isInitializing || (!isRunning && !isReadyToStep)}
              className="w-full gap-1.5 text-xs font-medium"
            >
              {isRunning ? (
                <>
                  <Pause className="h-3.5 w-3.5" /> Pause
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5" /> Auto Run
                </>
              )}
            </Button>

            <Button
              variant="outline"
              size="default"
              onClick={onStop}
              disabled={!isRunning && status !== "running"}
              className="w-full gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
            >
              <Square className="h-3.5 w-3.5" /> Stop
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
