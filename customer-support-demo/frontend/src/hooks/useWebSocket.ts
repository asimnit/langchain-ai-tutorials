import { useState, useEffect, useCallback, useRef } from 'react';
import type { WSEvent, ConnectionStatus } from '../types';

const WS_URL = 'ws://localhost:8000/ws';

interface UseWebSocketOptions {
  onEvent?: (event: WSEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { 
    onEvent, 
    reconnectAttempts = 5, 
    reconnectInterval = 3000 
  } = options;
  
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const clientIdRef = useRef<string>(`client-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const reconnectCountRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus('connecting');
    setError(null);

    const ws = new WebSocket(`${WS_URL}/${clientIdRef.current}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      reconnectCountRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSEvent;
        console.log('[WS] Received event:', data.type, data);
        onEvent?.(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    ws.onclose = () => {
      setStatus('disconnected');
      
      // Attempt to reconnect
      if (reconnectCountRef.current < reconnectAttempts) {
        reconnectCountRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectInterval);
      }
    };

    ws.onerror = () => {
      setStatus('error');
      setError('WebSocket connection failed');
    };
  }, [onEvent, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    reconnectCountRef.current = reconnectAttempts; // Prevent reconnection
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('disconnected');
  }, [reconnectAttempts]);

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'message',
        content
      }));
      return true;
    }
    return false;
  }, []);

  const clearHistory = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'clear' }));
      return true;
    }
    return false;
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    status,
    error,
    sendMessage,
    clearHistory,
    connect,
    disconnect,
    isConnected: status === 'connected'
  };
}
