export interface DOMRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DOMElement {
  index: string;
  role: string;
  label: string;
  value?: string;
  checked?: string;
  selected?: string;
  expanded?: string;
  operations: string[];
  rect?: DOMRect;
  options?: Array<{ index: string; label: string; value: string }>;
}

export interface Decision {
  choice: string;
  operation: string;
  target?: string;
  confidence: number;
  latency_ms: number;
  top_p?: Record<string, number>;
  answers?: Record<string, any>;
}

export interface HistoryItem {
  step: number;
  action: string;
  kind?: string;
  choice?: string;
  operation?: string;
  target?: string;
  confidence?: number;
  latency_ms?: number;
  text?: string | null;
  text_meta?: Record<string, any> | null;
  url?: string;
  elapsed_ms?: number;
  page_changed?: boolean;
}

export interface AgentState {
  status: 'idle' | 'ready' | 'predicted' | 'running' | 'done' | 'blocked' | 'error';
  url: string;
  goal: string;
  page_title?: string;
  page_url?: string;
  decision?: Decision | null;
  history: HistoryItem[];
  elements?: DOMElement[];
  step_count: number;
  elapsed_ms: number;
  screenshot?: string;
  error?: string;
  model?: string;
  device?: string;
}

export interface SystemInfo {
  model: string;
  device: string;
  host: string;
  port: number;
  headless: boolean;
  version: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  category: string;
  params?: string;
  description?: string;
  cached?: boolean;
  repo_url?: string;
}

export interface ModelStatus {
  status: 'idle' | 'loading' | 'success' | 'error';
  stage?: string;
  progress?: number;
  model?: string;
  device?: string;
  error?: string | null;
  elapsed_ms?: number;
}

export interface ToastItem {
  id: string;
  type: 'loading' | 'success' | 'error' | 'info';
  title: string;
  message?: string;
  stage?: string;
  progress?: number; // 0.0 to 1.0
  elapsed_ms?: number;
  device?: string;
  duration?: number; // auto-dismiss delay in ms
}
