import React, { useState, useEffect, useCallback } from "react";
import {
  Cpu,
  RefreshCw,
  ChevronDown,
  Check,
  Download,
  Zap,
  Sparkles,
  Layers,
  Plus,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { Input } from "./ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { ModelInfo } from "@/types";
import { cn } from "@/lib/utils";

interface ModelSelectorProps {
  currentModel: string;
  currentDevice?: string;
  onSelectModel: (modelId: string) => Promise<void>;
  isLoading: boolean;
  loadingStage?: string;
  loadingProgress?: number;
  onRefreshModels?: () => Promise<void>;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  currentModel,
  currentDevice = "MPS",
  onSelectModel,
  isLoading,
  loadingStage,
  loadingProgress,
  onRefreshModels,
}) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const [customModelInput, setCustomModelInput] = useState("");
  const [customError, setCustomError] = useState<string | null>(null);

  // Fetch models list from API
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        const data = await res.json();
        if (data.models && Array.isArray(data.models)) {
          setModels(data.models);
        }
      }
    } catch (e) {
      console.warn("Failed to fetch available models:", e);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleRefresh = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsRefreshing(true);
    try {
      const res = await fetch("/api/models/refresh", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        if (data.models) {
          setModels(data.models);
        }
      }
      if (onRefreshModels) {
        await onRefreshModels();
      }
    } catch (e) {
      console.warn("Failed to refresh models:", e);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleCustomSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const target = customModelInput.trim();
    if (!target) return;

    setCustomError(null);
    try {
      setCustomModalOpen(false);
      setCustomModelInput("");
      await onSelectModel(target);
      fetchModels();
    } catch (err: any) {
      setCustomError(err.message || String(err));
    }
  };

  // Find active model info
  const activeModelInfo = models.find(
    (m) => m.id === currentModel || m.name === currentModel
  );

  const displayName = activeModelInfo ? activeModelInfo.name : currentModel;

  // Group models by category
  const featuredModels = models.filter((m) => m.category === "Featured");
  const officialModels = models.filter((m) => m.category === "Official Checkpoints");
  const baseModels = models.filter((m) => m.category === "Base Models");
  const otherModels = models.filter(
    (m) => !["Featured", "Official Checkpoints", "Base Models"].includes(m.category)
  );

  return (
    <div className="flex items-center gap-1.5 font-mono text-xs">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            disabled={isLoading}
            className={cn(
              "flex items-center gap-2 px-2.5 py-1 rounded-md border border-border bg-secondary/40 text-foreground hover:bg-secondary/70 transition-colors focus:outline-none focus:ring-1 focus:ring-ring text-xs select-none max-w-[280px] sm:max-w-[340px]",
              isLoading && "border-sky-500/50 bg-sky-950/20 cursor-wait"
            )}
            title={`Active Checkpoint: ${displayName}`}
          >
            {isLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-sky-400 shrink-0" />
            ) : (
              <Cpu className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            )}

            <span className="truncate font-sans font-medium text-xs">
              {isLoading ? (
                <span className="text-sky-300 font-mono text-[11px]">
                  {loadingStage || "Hot-swapping model..."}
                </span>
              ) : (
                displayName
              )}
            </span>

            {!isLoading && (
              <span className="hidden sm:inline-block px-1.5 py-0.2 rounded border border-border bg-secondary text-[10px] text-muted-foreground uppercase font-mono shrink-0">
                {currentDevice}
              </span>
            )}

            <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0 ml-auto" />
          </button>
        </DropdownMenuTrigger>

        <DropdownMenuContent
          align="start"
          className="w-[360px] sm:w-[420px] max-h-[500px] overflow-y-auto p-1.5 bg-card/95 backdrop-blur-md border-border"
        >
          {/* Header Info */}
          <div className="px-2 py-1.5 mb-1 border-b border-border flex items-center justify-between">
            <span className="text-[11px] font-semibold tracking-wider uppercase text-muted-foreground font-sans">
              Model Checkpoints
            </span>
          </div>

          {/* Featured Models */}
          {featuredModels.length > 0 && (
            <>
              <DropdownMenuLabel className="flex items-center gap-1.5 text-sky-400">
                <Sparkles className="h-3 w-3" /> New Checkpoints
              </DropdownMenuLabel>
              {featuredModels.map((m) => (
                <ModelMenuItem
                  key={m.id}
                  model={m}
                  isSelected={currentModel === m.id || displayName === m.name}
                  onSelect={() => onSelectModel(m.id)}
                />
              ))}
              <DropdownMenuSeparator />
            </>
          )}

          {/* Official Models */}
          {officialModels.length > 0 && (
            <>
              <DropdownMenuLabel className="flex items-center gap-1.5 text-muted-foreground">
                <Zap className="h-3 w-3" /> Official Checkpoints
              </DropdownMenuLabel>
              {officialModels.map((m) => (
                <ModelMenuItem
                  key={m.id}
                  model={m}
                  isSelected={currentModel === m.id || displayName === m.name}
                  onSelect={() => onSelectModel(m.id)}
                />
              ))}
              <DropdownMenuSeparator />
            </>
          )}

          {/* Other / Cached Checkpoints */}
          {otherModels.length > 0 && (
            <>
              <DropdownMenuLabel className="flex items-center gap-1.5 text-muted-foreground">
                <Layers className="h-3 w-3" /> Cached & Discovered Models
              </DropdownMenuLabel>
              {otherModels.map((m) => (
                <ModelMenuItem
                  key={m.id}
                  model={m}
                  isSelected={currentModel === m.id || displayName === m.name}
                  onSelect={() => onSelectModel(m.id)}
                />
              ))}
              <DropdownMenuSeparator />
            </>
          )}

          {/* Base Models */}
          {baseModels.length > 0 && (
            <>
              <DropdownMenuLabel className="flex items-center gap-1.5 text-muted-foreground">
                Foundation & Base
              </DropdownMenuLabel>
              {baseModels.map((m) => (
                <ModelMenuItem
                  key={m.id}
                  model={m}
                  isSelected={currentModel === m.id || displayName === m.name}
                  onSelect={() => onSelectModel(m.id)}
                />
              ))}
              <DropdownMenuSeparator />
            </>
          )}

          {/* Custom Model Input Option */}
          <DropdownMenuItem
            onSelect={() => setCustomModalOpen(true)}
            className="flex items-center gap-2 py-2 text-foreground font-sans font-medium text-xs cursor-pointer hover:bg-secondary"
          >
            <Plus className="h-3.5 w-3.5 text-primary" />
            <span>Load Custom HuggingFace Model...</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              onClick={handleRefresh}
              disabled={isLoading || isRefreshing}
              className="h-7 w-7 text-muted-foreground hover:text-foreground border-border bg-secondary/30"
            >
              <RefreshCw
                className={cn("h-3 w-3", isRefreshing && "animate-spin text-foreground")}
              />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Scan & refresh checkpoints</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Custom Model Modal */}
      <Dialog open={customModalOpen} onOpenChange={setCustomModalOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
              <Cpu className="h-4 w-4" /> Load Custom Model Checkpoint
            </DialogTitle>
            <DialogDescription>
              Enter any Hugging Face repo ID, checkpoint URL, or local path to hot-swap in-memory.
            </DialogDescription>
          </DialogHeader>

          <form onSubmit={handleCustomSubmit} className="space-y-4 pt-2">
            <div className="space-y-2">
              <label className="text-xs font-medium text-muted-foreground">
                Hugging Face Repo ID, URL, or Path
              </label>
              <Input
                value={customModelInput}
                onChange={(e) => setCustomModelInput(e.target.value)}
                placeholder="e.g. ichenney/laya-browser-v32b or https://huggingface.co/..."
                className="font-mono text-xs"
                autoFocus
              />
            </div>

            {/* Quick Suggestions Chips */}
            <div className="space-y-1.5">
              <div className="text-[11px] text-muted-foreground">Examples & suggestions:</div>
              <div className="flex flex-wrap gap-1.5">
                {[
                  "https://huggingface.co/ichenney/laya-browser-v32b",
                  "https://huggingface.co/abedinia/laya-web-agent",
                  "v10s",
                ].map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    onClick={() => setCustomModelInput(chip)}
                    className="px-2 py-0.5 rounded border border-border bg-secondary/50 text-[10px] font-mono text-muted-foreground hover:text-foreground hover:bg-secondary truncate max-w-[320px]"
                  >
                    {chip.replace("https://huggingface.co/", "")}
                  </button>
                ))}
              </div>
            </div>

            {customError && (
              <div className="text-xs text-rose-400 bg-rose-950/20 border border-rose-900/30 p-2 rounded">
                {customError}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setCustomModalOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="default"
                size="sm"
                disabled={!customModelInput.trim()}
              >
                Hot-Swap Model
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
};

const ModelMenuItem: React.FC<{
  model: ModelInfo;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ model, isSelected, onSelect }) => {
  return (
    <DropdownMenuItem
      onSelect={onSelect}
      className={cn(
        "flex items-start justify-between gap-2 py-2 px-2.5 rounded cursor-pointer transition-colors font-sans hover:bg-secondary",
        isSelected && "bg-secondary/60 text-foreground"
      )}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-xs text-foreground truncate">
            {model.name}
          </span>
          {model.params && (
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded border border-border bg-secondary/50 text-muted-foreground">
              {model.params}
            </span>
          )}
        </div>

        {model.description && (
          <p className="text-[11px] text-muted-foreground leading-snug line-clamp-1 mt-0.5">
            {model.description}
          </p>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0 pt-0.5">
        {model.cached ? (
          <span
            className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded border border-emerald-900/40 bg-emerald-950/30 text-emerald-400"
            title="Model cached locally (Instant Hot-Swap)"
          >
            <Zap className="h-2.5 w-2.5" /> Cached
          </span>
        ) : (
          <span
            className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded border border-sky-900/40 bg-sky-950/30 text-sky-400"
            title="Will download snapshot from Hugging Face Hub"
          >
            <Download className="h-2.5 w-2.5" /> HF
          </span>
        )}

        {isSelected && <Check className="h-3.5 w-3.5 text-foreground shrink-0" />}
      </div>
    </DropdownMenuItem>
  );
};
