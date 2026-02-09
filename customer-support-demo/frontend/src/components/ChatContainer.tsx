import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageSquare, Trash2, Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ThinkingIndicator } from './ThinkingIndicator';
import type { ChatMessage as ChatMessageType, WSEvent, ToolExecution, ConnectionStatus } from '../types';
import clsx from 'clsx';

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function ChatContainer() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentThinking, setCurrentThinking] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentAssistantIdRef = useRef<string | null>(null);
  
  // Pending assistant message ref - source of truth for in-progress message
  const pendingAssistantRef = useRef<{
    id: string;
    toolExecutions: ToolExecution[];
    content: string;
    thinking: string | null;
  } | null>(null);

  // Sync pending ref to state (idempotent - safe to call multiple times)
  const syncPendingToState = useCallback(() => {
    const pending = pendingAssistantRef.current;
    if (!pending) return;

    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === pending.id);
      const msg: ChatMessageType = {
        id: pending.id,
        role: 'assistant',
        content: pending.content,
        timestamp: new Date(),
        toolExecutions: pending.toolExecutions.length > 0
          ? [...pending.toolExecutions]
          : undefined,
        isStreaming: true,
        thinking: pending.thinking || undefined,
      };

      if (idx >= 0) {
        // Replace existing
        const updated = [...prev];
        updated[idx] = msg;
        return updated;
      }
      // Add new
      return [...prev, msg];
    });
  }, []);

  // Scroll to bottom when messages change
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentThinking, scrollToBottom]);

  // Handle WebSocket events
  const handleWSEvent = useCallback((event: WSEvent) => {
    console.log('[Chat] Handling event:', event.type);
    switch (event.type) {
      case 'thinking':
        // If we have a pending message, add thinking to it
        if (pendingAssistantRef.current) {
          pendingAssistantRef.current.thinking = event.content;
          syncPendingToState();
        } else {
          // No pending message yet, use standalone thinking
          setCurrentThinking(event.content);
        }
        break;

      case 'tool_call': {
        setCurrentThinking(null);
        const toolExecution: ToolExecution = {
          id: generateId(),
          name: event.name,
          args: event.args,
          status: 'calling',
        };

        // Initialize pending if needed
        if (!pendingAssistantRef.current) {
          const newId = generateId();
          pendingAssistantRef.current = {
            id: newId,
            toolExecutions: [],
            content: '',
            thinking: null,
          };
          currentAssistantIdRef.current = newId;
          setCurrentThinking(null);  // Clear standalone thinking
          console.log('[Chat] tool_call - created pending assistant:', newId);
        }

        // Add tool to ref (synchronous)
        pendingAssistantRef.current.toolExecutions.push(toolExecution);
        console.log('[Chat] tool_call - added tool:', event.name, 'total tools:', pendingAssistantRef.current.toolExecutions.length);

        // Sync to state
        syncPendingToState();
        break;
      }

      case 'tool_result': {
        // Update tool in ref (synchronous)
        if (pendingAssistantRef.current) {
          pendingAssistantRef.current.toolExecutions = pendingAssistantRef.current.toolExecutions.map((te) =>
            te.name === event.name && te.status === 'calling'
              ? { ...te, result: event.result, status: 'completed' as const }
              : te
          );
          console.log('[Chat] tool_result - updated tool:', event.name);

          // Sync to state
          syncPendingToState();
        }
        break;
      }

      case 'response': {
        setCurrentThinking(null);

        // Initialize pending if needed
        if (!pendingAssistantRef.current) {
          const newId = generateId();
          pendingAssistantRef.current = {
            id: newId,
            toolExecutions: [],
            content: '',
            thinking: null,
          };
          currentAssistantIdRef.current = newId;
          setCurrentThinking(null);  // Clear standalone thinking
          console.log('[Chat] response - created pending assistant:', newId);
        }

        // Clear thinking when content starts arriving
        pendingAssistantRef.current.thinking = null;

        // Append content to ref (synchronous)
        pendingAssistantRef.current.content += event.content;
        console.log('[Chat] response - content length:', pendingAssistantRef.current.content.length, 'tools:', pendingAssistantRef.current.toolExecutions.length);

        // Sync to state
        syncPendingToState();
        break;
      }

      case 'done':
        setIsLoading(false);
        setCurrentThinking(null);
        
        // Mark current message as not streaming
        setMessages((prev) => {
          console.log('[Chat] Done - final messages count:', prev.length);
          if (pendingAssistantRef.current) {
            console.log('[Chat] Done - pending tools:', pendingAssistantRef.current.toolExecutions.length);
          }
          return prev.map((msg) =>
            msg.isStreaming ? { ...msg, isStreaming: false } : msg
          );
        });
        
        // Clear pending ref for next turn
        pendingAssistantRef.current = null;
        break;

      case 'error':
        setIsLoading(false);
        setCurrentThinking(null);
        currentAssistantIdRef.current = null;
        pendingAssistantRef.current = null;
        
        // Add error message
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            role: 'assistant',
            content: `Error: ${event.content}`,
            timestamp: new Date(),
          },
        ]);
        break;

      case 'cleared':
        setMessages([]);
        pendingAssistantRef.current = null;
        break;
    }
  }, []);

  const { status, sendMessage, clearHistory, isConnected } = useWebSocket({
    onEvent: handleWSEvent,
  });

  const handleSend = useCallback(
    (content: string) => {
      if (!isConnected || isLoading) return;

      // Add user message
      const userMessage: ChatMessageType = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Send to backend
      setIsLoading(true);
      currentAssistantIdRef.current = null;
      pendingAssistantRef.current = null;  // Clear pending for new turn
      sendMessage(content);
    },
    [isConnected, isLoading, sendMessage]
  );

  const handleClear = useCallback(() => {
    clearHistory();
    setMessages([]);
  }, [clearHistory]);

  return (
    <div className="flex flex-col h-screen bg-gray-900">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">Customer Support</h1>
            <div className="flex items-center gap-2 text-xs">
              <StatusIndicator status={status} />
            </div>
          </div>
        </div>

        <button
          onClick={handleClear}
          disabled={messages.length === 0}
          className={clsx(
            'flex items-center gap-2 px-3 py-2 rounded-lg',
            'text-gray-400 hover:text-white hover:bg-gray-800',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-colors duration-200'
          )}
        >
          <Trash2 className="w-4 h-4" />
          <span className="text-sm">Clear</span>
        </button>
      </header>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-6 chat-scrollbar">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <MessageSquare className="w-12 h-12 mb-4 opacity-50" />
            <p className="text-lg font-medium">How can I help you today?</p>
            <p className="text-sm mt-1">Ask about orders, returns, shipping, and more.</p>
            
            {/* Suggested prompts */}
            <div className="mt-6 flex flex-wrap gap-2 justify-center max-w-md">
              {[
                'What\'s your return policy?',
                'Track order ORD-001',
                'Shipping options?',
              ].map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => handleSend(prompt)}
                  disabled={!isConnected}
                  className="px-3 py-1.5 rounded-full border border-gray-700 text-sm text-gray-400 hover:text-white hover:border-indigo-500 transition-colors disabled:opacity-50"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="py-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}

            {/* Current thinking indicator */}
            {currentThinking && (
              <div className="flex gap-3 py-4">
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
                  <MessageSquare className="w-4 h-4 text-indigo-300" />
                </div>
                <ThinkingIndicator message={currentThinking} />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <ChatInput
        onSend={handleSend}
        disabled={!isConnected}
        isLoading={isLoading}
        placeholder={
          isConnected
            ? 'Ask about orders, shipping, returns...'
            : 'Connecting...'
        }
      />
    </div>
  );
}

function StatusIndicator({ status }: { status: ConnectionStatus }) {
  switch (status) {
    case 'connected':
      return (
        <span className="flex items-center gap-1 text-green-400">
          <Wifi className="w-3 h-3" />
          Connected
        </span>
      );
    case 'connecting':
      return (
        <span className="flex items-center gap-1 text-yellow-400">
          <Wifi className="w-3 h-3 animate-pulse" />
          Connecting...
        </span>
      );
    case 'disconnected':
      return (
        <span className="flex items-center gap-1 text-gray-500">
          <WifiOff className="w-3 h-3" />
          Disconnected
        </span>
      );
    case 'error':
      return (
        <span className="flex items-center gap-1 text-red-400">
          <AlertCircle className="w-3 h-3" />
          Connection error
        </span>
      );
  }
}
