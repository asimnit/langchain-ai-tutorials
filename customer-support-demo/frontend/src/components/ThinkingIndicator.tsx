import { Sparkles } from 'lucide-react';

interface ThinkingIndicatorProps {
  message?: string;
}

export function ThinkingIndicator({ message = 'Thinking' }: ThinkingIndicatorProps) {
  return (
    <div className="flex items-center gap-2 text-gray-400 py-2 message-fade-in">
      <Sparkles className="w-4 h-4 text-indigo-400 animate-pulse" />
      <span className="text-sm">{message}</span>
      <div className="flex gap-1">
        <span className="thinking-dot w-1.5 h-1.5 bg-indigo-400 rounded-full" />
        <span className="thinking-dot w-1.5 h-1.5 bg-indigo-400 rounded-full" />
        <span className="thinking-dot w-1.5 h-1.5 bg-indigo-400 rounded-full" />
      </div>
    </div>
  );
}
