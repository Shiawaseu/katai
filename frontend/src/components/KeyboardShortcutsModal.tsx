import React from "react";
import { Keyboard } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";

interface KeyboardShortcutsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  open,
  onOpenChange,
}) => {
  const shortcuts = [
    { key: "Space", desc: "Execute single predict-and-act step" },
    { key: "R", desc: "Toggle Auto-Run mode on / off" },
    { key: "Esc", desc: "Stop active execution or close modal" },
    { key: "I", desc: "Initialize browser agent" },
    { key: "?", desc: "Toggle keyboard shortcuts help" },
  ];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-sm font-semibold">
            <Keyboard className="h-4 w-4 text-muted-foreground" />
            Keyboard Shortcuts
          </DialogTitle>
          <DialogDescription>
            Keyboard shortcuts for rapid operation.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2 py-2">
          {shortcuts.map((s) => (
            <div
              key={s.key}
              className="flex items-center justify-between p-2 rounded-md bg-secondary/40 border border-border text-xs"
            >
              <span className="text-foreground/90 font-medium">{s.desc}</span>
              <kbd className="px-2 py-0.5 rounded bg-secondary border border-border font-mono text-xs font-medium text-foreground">
                {s.key}
              </kbd>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};
