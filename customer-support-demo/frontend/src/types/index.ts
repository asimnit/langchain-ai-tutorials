// WebSocket event types from the backend
export type WSEventType = 
  | 'thinking' 
  | 'tool_call' 
  | 'tool_result' 
  | 'response' 
  | 'done' 
  | 'error' 
  | 'cleared'
  | 'pong';

export interface WSThinkingEvent {
  type: 'thinking';
  content: string;
}

export interface WSToolCallEvent {
  type: 'tool_call';
  name: string;
  args: Record<string, unknown>;
}

export interface WSToolResultEvent {
  type: 'tool_result';
  name: string;
  result: string;
}

export interface WSResponseEvent {
  type: 'response';
  content: string;
  done: boolean;
}

export interface WSDoneEvent {
  type: 'done';
  messages: { role: string; content: string }[];
}

export interface WSErrorEvent {
  type: 'error';
  content: string;
}

export interface WSClearedEvent {
  type: 'cleared';
}

export interface WSPongEvent {
  type: 'pong';
}

export type WSEvent = 
  | WSThinkingEvent
  | WSToolCallEvent
  | WSToolResultEvent
  | WSResponseEvent
  | WSDoneEvent
  | WSErrorEvent
  | WSClearedEvent
  | WSPongEvent;

// Chat message types for the UI
export type MessageRole = 'user' | 'assistant';

export interface ToolExecution {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: 'calling' | 'completed';
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  toolExecutions?: ToolExecution[];
  thinking?: string;
}

// Connection status
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';
