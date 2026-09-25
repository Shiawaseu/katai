import React, { useState, useEffect, useRef, useCallback } from "react";
import { Header } from "./components/Header";
import { TaskConfig } from "./components/TaskConfig";
import { LiveViewport } from "./components/LiveViewport";
import { ActionSpaceTable } from "./components/ActionSpaceTable";
import { DecisionMetrics } from "./components/DecisionMetrics";
import { ExecutionTrace } from "./components/ExecutionTrace";
import { RawStateViewer } from "./components/RawStateViewer";
import { KeyboardShortcutsModal } from "./components/KeyboardShortcutsModal";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { ToastProvider, useToast } from "./components/ui/toast";
import { Eye, Layers, Code2, AlertCircle } from "lucide-react";
import { AgentState, DOMElement, HistoryItem } from "./types";

const INITIAL_STATE: AgentState = {
  status: "idle",
  url: "https://en.wikipedia.org/wiki/Main_Page",
  goal: "Search Wikipedia for 'Python programming language' and open the article about the Python language.",
  step_count: 0,
  elapsed_ms: 0,
  history: [],
  elements: [],
};

const MainConsole: React.FC = () => {
  const { addToast, updateToast } = useToast();
  const [state, setState] = useState<AgentState>(INITIAL_STATE);
  const [url, setUrl] = useState<string>("https://en.wikipedia.org/wiki/Main_Page");
  const [goal, setGoal] = useState<string>(
    "Search Wikipedia for 'Python programming language' and open the article about the Python language."
  );
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [isStepping, setIsStepping] = useState<boolean>(false);
  const [stepDelay, setStepDelay] = useState<number>(300);
  const [activeTab, setActiveTab] = useState<string>("live");
  const [hoveredElementIndex, setHoveredElementIndex] = useState<string | null>(null);
  const [selectedElementIndex, setSelectedElementIndex] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);

  // Model Hot-Swapping State
  const [isModelLoading, setIsModelLoading] = useState<boolean>(false);
  const [modelLoadingStage, setModelLoadingStage] = useState<string>("");
  const [modelLoadingProgress, setModelLoadingProgress] = useState<number>(0);

  const isRunningRef = useRef(isRunning);
  isRunningRef.current = isRunning;

  // Fetch initial agent and model state
  const fetchState = useCallback(async () => {
    try {
      const res = await fetch("/api/agent/state");
      if (res.ok) {
        const data = await res.json();
        setState((prev) => {
          if (data.url && !prev.url) setUrl(data.url);
          if (data.goal && !prev.goal) setGoal(data.goal);
          return {
            ...prev,
            ...data,
          };
        });
      }
    } catch (err) {
      console.warn("Could not fetch agent state:", err);
    }
  }, []);

  useEffect(() => {
    fetchState();
  }, [fetchState]);

  const handleSelectModel = async (modelId: string) => {
    setIsModelLoading(true);
    setModelLoadingStage("Resolving checkpoint...");
    setModelLoadingProgress(0.1);

    const toastId = "model-hot-swap";
    addToast({
      id: toastId,
      type: "loading",
      title: "Loading Model Checkpoint",
      message: `Hot-swapping to ${modelId}...`,
      stage: "Connecting to backend...",
      progress: 0.1,
    });

    // Real-time status polling for progress transitions
    let pollInterval: any = null;
    pollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/models/status");
        if (res.ok) {
          const status = await res.json();
          if (status.status === "loading") {
            if (status.stage) setModelLoadingStage(status.stage);
            if (status.progress !== undefined) setModelLoadingProgress(status.progress);
            updateToast(toastId, {
              stage: status.stage,
              progress: status.progress,
            });
          }
        }
      } catch {}
    }, 250);

    try {
      const res = await fetch("/api/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelId }),
      });

      clearInterval(pollInterval);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Failed to load model: ${res.statusText}`);
      }

      const data = await res.json();
      setState((prev) => ({
        ...prev,
        model: data.checkpoint || modelId,
        device: data.device || prev.device,
      }));

      updateToast(toastId, {
        type: "success",
        title: "Checkpoint Loaded",
        message: `Successfully hot-swapped to ${modelId}`,
        elapsed_ms: data.elapsed_ms,
        device: data.device,
        progress: 1.0,
        duration: 4500,
      });
    } catch (err: any) {
      clearInterval(pollInterval);
      const errMsg = err.message || String(err);
      updateToast(toastId, {
        type: "error",
        title: "Checkpoint Load Failed",
        message: errMsg,
      });
      setErrorMessage(`Failed to hot-swap model: ${errMsg}`);
    } finally {
      setIsModelLoading(false);
      setModelLoadingStage("");
      setModelLoadingProgress(0);
    }
  };

  const handleRefreshModels = async () => {
    addToast({
      type: "info",
      title: "Checkpoints Rescanned",
      message: "Rescanned local HF cache and models directory.",
      duration: 3000,
    });
  };

  // Start / Initialize Agent
  const handleInitialize = async () => {
    setIsInitializing(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/agent/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, goal, checkpoint: state.model }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to initialize agent");
      }
      const data = await res.json();
      setState((prev) => ({ ...prev, ...data }));
    } catch (err: any) {
      setErrorMessage(err.message || String(err));
    } finally {
      setIsInitializing(false);
    }
  };

  // Execute single step
  const handleStep = async (): Promise<AgentState | null> => {
    setIsStepping(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/agent/step", { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Failed to execute step");
      }
      const data = await res.json();
      setState((prev) => ({ ...prev, ...data }));
      return data;
    } catch (err: any) {
      setErrorMessage(err.message || String(err));
      return null;
    } finally {
      setIsStepping(false);
    }
  };

  // Toggle Auto-Run
  const handleToggleRun = async () => {
    if (isRunning) {
      setIsRunning(false);
      return;
    }

    setIsRunning(true);
    setErrorMessage(null);

    while (isRunningRef.current) {
      const updated = await handleStep();
      if (!updated || updated.status === "done" || updated.status === "blocked") {
        setIsRunning(false);
        break;
      }
      await new Promise((r) => setTimeout(r, stepDelay));
    }
  };

  // Stop Agent
  const handleStop = async () => {
    setIsRunning(false);
    try {
      await fetch("/api/agent/stop", { method: "POST" });
    } catch {}
    setState((prev) => ({
      ...prev,
      status: prev.status === "running" ? "ready" : prev.status,
    }));
  };

  // Reset Session
  const handleReset = async () => {
    setIsRunning(false);
    setErrorMessage(null);
    try {
      await fetch("/api/agent/reset", { method: "POST" });
    } catch {}
    setState({
      ...INITIAL_STATE,
      url,
      goal,
      model: state.model,
      device: state.device,
    });
  };

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        if (!isRunning && !isInitializing && !isModelLoading && (state.status === "ready" || state.status === "predicted")) {
          handleStep();
        }
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        handleToggleRun();
      } else if (e.key === "Escape") {
        e.preventDefault();
        if (shortcutsOpen) {
          setShortcutsOpen(false);
        } else {
          handleStop();
        }
      } else if (e.key === "?") {
        e.preventDefault();
        setShortcutsOpen(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isRunning, isInitializing, isModelLoading, state.status, shortcutsOpen]);

  const lastHistoryItem = state.history.length > 0 ? state.history[state.history.length - 1] : null;

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground antialiased selection:bg-secondary">
      <Header
        state={state}
        onReset={handleReset}
        onOpenShortcuts={() => setShortcutsOpen(true)}
        isBusy={isInitializing || isStepping || isRunning || isModelLoading}
        onSelectModel={handleSelectModel}
        isModelLoading={isModelLoading}
        modelLoadingStage={modelLoadingStage}
        modelLoadingProgress={modelLoadingProgress}
        onRefreshModels={handleRefreshModels}
      />

      {/* Error Alert */}
      {errorMessage && (
        <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-2 flex items-center justify-between text-xs text-red-300">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-muted-foreground hover:text-foreground text-xs font-medium px-2 py-0.5"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Grid */}
      <main className="flex-1 p-3 lg:p-4 grid grid-cols-1 lg:grid-cols-12 gap-3 lg:gap-4 overflow-hidden max-w-[1920px] w-full mx-auto">
        {/* Left Column: Task Config & Decision Metrics */}
        <section className="lg:col-span-4 xl:col-span-3 flex flex-col gap-3 overflow-y-auto">
          <TaskConfig
            url={url}
            setUrl={setUrl}
            goal={goal}
            setGoal={setGoal}
            status={state.status}
            isRunning={isRunning}
            onInitialize={handleInitialize}
            onStep={handleStep}
            onToggleRun={handleToggleRun}
            onStop={handleStop}
            stepDelay={stepDelay}
            setStepDelay={setStepDelay}
            isInitializing={isInitializing || isModelLoading}
          />

          <DecisionMetrics
            decision={state.decision}
            lastAction={lastHistoryItem}
          />
        </section>

        {/* Center Column: Live Viewport & Action Space Tabs */}
        <section className="lg:col-span-5 xl:col-span-6 flex flex-col h-[750px] lg:h-[calc(100vh-80px)] min-h-[500px]">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-col h-full">
            <div className="flex items-center justify-between pb-2">
              <TabsList className="bg-secondary/40 border border-border">
                <TabsTrigger value="live" className="gap-1.5 text-xs">
                  <Eye className="h-3.5 w-3.5" />
                  Live Viewport
                </TabsTrigger>
                <TabsTrigger value="dom" className="gap-1.5 text-xs">
                  <Layers className="h-3.5 w-3.5" />
                  DOM Controls
                  {state.elements && state.elements.length > 0 && (
                    <span className="ml-1 text-[10px] px-1.5 py-0.2 rounded bg-secondary text-foreground font-mono font-medium border border-border">
                      {state.elements.length}
                    </span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="raw" className="gap-1.5 text-xs">
                  <Code2 className="h-3.5 w-3.5" />
                  State JSON
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="live" className="flex-1 mt-0 h-full overflow-hidden">
              <LiveViewport
                screenshot={state.screenshot}
                url={state.page_url || state.url}
                title={state.page_title}
                elements={state.elements}
                decision={state.decision}
                hoveredElementIndex={hoveredElementIndex}
                setHoveredElementIndex={setHoveredElementIndex}
                onRefresh={fetchState}
                onLaunch={handleInitialize}
                isInitializing={isInitializing || isModelLoading}
              />
            </TabsContent>

            <TabsContent value="dom" className="flex-1 mt-0 h-full overflow-hidden">
              <ActionSpaceTable
                elements={state.elements || []}
                hoveredElementIndex={hoveredElementIndex}
                setHoveredElementIndex={setHoveredElementIndex}
                selectedElementIndex={selectedElementIndex}
                setSelectedElementIndex={setSelectedElementIndex}
              />
            </TabsContent>

            <TabsContent value="raw" className="flex-1 mt-0 h-full overflow-hidden">
              <RawStateViewer state={state} />
            </TabsContent>
          </Tabs>
        </section>

        {/* Right Column: Execution History Trace */}
        <section className="lg:col-span-3 xl:col-span-3 flex flex-col h-[600px] lg:h-[calc(100vh-80px)] overflow-hidden">
          <ExecutionTrace
            history={state.history}
            onSelectStep={(step) => {
              if (step.choice) {
                const cleanIdx = step.choice.replace(/^e/, "");
                setSelectedElementIndex(cleanIdx);
                setHoveredElementIndex(cleanIdx);
              }
            }}
          />
        </section>
      </main>

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        open={shortcutsOpen}
        onOpenChange={setShortcutsOpen}
      />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <ToastProvider>
      <MainConsole />
    </ToastProvider>
  );
};

export default App;
