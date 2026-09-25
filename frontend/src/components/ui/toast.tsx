import React, { createContext, useContext, useState, useCallback, useEffect } from "react";
import { CheckCircle2, AlertCircle, Loader2, Info, X } from "lucide-react";
import { ToastItem } from "@/types";
import { Progress } from "./progress";
import { cn } from "@/lib/utils";

interface ToastContextType {
  toasts: ToastItem[];
  addToast: (toast: Omit<ToastItem, "id"> & { id?: string }) => string;
  updateToast: (id: string, updates: Partial<ToastItem>) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback((toast: Omit<ToastItem, "id"> & { id?: string }) => {
    const id = toast.id || `toast-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const newToast: ToastItem = {
      ...toast,
      id,
    };

    setToasts((prev) => {
      // If a toast with this id already exists, replace it
      const exists = prev.some((t) => t.id === id);
      if (exists) {
        return prev.map((t) => (t.id === id ? newToast : t));
      }
      return [...prev, newToast];
    });

    if (newToast.duration && newToast.duration > 0) {
      setTimeout(() => {
        removeToast(id);
      }, newToast.duration);
    }

    return id;
  }, [removeToast]);

  const updateToast = useCallback((id: string, updates: Partial<ToastItem>) => {
    setToasts((prev) =>
      prev.map((t) => {
        if (t.id !== id) return t;
        const updated = { ...t, ...updates };
        if (updated.duration && updated.duration > 0 && !t.duration) {
          setTimeout(() => {
            removeToast(id);
          }, updated.duration);
        }
        return updated;
      })
    );
  }, [removeToast]);

  return (
    <ToastContext.Provider value={{ toasts, addToast, updateToast, removeToast }}>
      {children}
      {/* Toast Viewport Floating Container */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2.5 max-w-md w-full pointer-events-none px-4 sm:px-0">
        {toasts.map((toast) => (
          <ToastCard key={toast.id} toast={toast} onDismiss={() => removeToast(toast.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
};

const ToastCard: React.FC<{ toast: ToastItem; onDismiss: () => void }> = ({ toast, onDismiss }) => {
  const getIcon = () => {
    switch (toast.type) {
      case "loading":
        return <Loader2 className="h-4 w-4 animate-spin text-sky-400 shrink-0 mt-0.5" />;
      case "success":
        return <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />;
      default:
        return <Info className="h-4 w-4 text-primary shrink-0 mt-0.5" />;
    }
  };

  const getBorderColor = () => {
    switch (toast.type) {
      case "loading":
        return "border-sky-500/40 shadow-sky-950/20";
      case "success":
        return "border-emerald-500/40 shadow-emerald-950/20";
      case "error":
        return "border-rose-500/40 shadow-rose-950/20";
      default:
        return "border-border";
    }
  };

  const percentage = Math.min(Math.max(Math.round((toast.progress || 0) * 100), 0), 100);

  return (
    <div
      role="alert"
      className={cn(
        "pointer-events-auto w-full rounded-lg border bg-card/95 backdrop-blur-md p-3.5 shadow-lg transition-all duration-300 animate-in slide-in-from-bottom-2 fade-in-0",
        getBorderColor()
      )}
    >
      <div className="flex items-start justify-between gap-2.5">
        <div className="flex items-start gap-2.5 flex-1 min-w-0">
          {getIcon()}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold tracking-tight text-foreground font-sans">
                {toast.title}
              </span>
              {toast.device && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded border border-border bg-secondary/60 text-muted-foreground uppercase">
                  {toast.device}
                </span>
              )}
              {toast.elapsed_ms !== undefined && toast.elapsed_ms > 0 && (
                <span className="text-[10px] font-mono text-muted-foreground">
                  {(toast.elapsed_ms / 1000).toFixed(1)}s
                </span>
              )}
            </div>

            {toast.message && (
              <p className="mt-1 text-xs text-muted-foreground break-words leading-relaxed">
                {toast.message}
              </p>
            )}

            {toast.stage && (
              <p className="mt-1 text-[11px] font-mono text-sky-300/90 truncate">
                {toast.stage}
              </p>
            )}

            {toast.type === "loading" && toast.progress !== undefined && (
              <div className="mt-2.5 space-y-1">
                <Progress
                  value={percentage}
                  max={100}
                  className="h-1.5 bg-secondary"
                  indicatorClassName="bg-sky-400"
                />
                <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
                  <span>Progress</span>
                  <span className="text-sky-300 font-medium">{percentage}%</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <button
          onClick={onDismiss}
          type="button"
          className="text-muted-foreground/60 hover:text-foreground p-0.5 rounded transition-colors"
          title="Dismiss notification"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
};
