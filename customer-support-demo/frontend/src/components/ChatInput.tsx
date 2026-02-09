import { useState, useRef, useEffect } from 'react';
import type { KeyboardEvent } from 'react';
import { Send, Loader2 } from 'lucide-react';
import clsx from 'clsx';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  isLoading = false,
  placeholder = 'Type your message...',
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleSend = () => {
    const trimmed = value.trim();
    if (trimmed && !disabled && !isLoading) {
      onSend(trimmed);
      setValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex items-end gap-2 p-4 border-t border-gray-800 bg-gray-900">
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          rows={1}
          className={clsx(
            'w-full resize-none rounded-xl px-4 py-3 pr-12',
            'bg-gray-800 border border-gray-700',
            'text-gray-100 placeholder-gray-500',
            'focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent',
            'disabled:opacity-50 disabled:cursor-not-allowed',
            'transition-all duration-200'
          )}
        />
      </div>

      <button
        onClick={handleSend}
        disabled={!value.trim() || disabled || isLoading}
        className={clsx(
          'flex-shrink-0 w-10 h-10 rounded-xl',
          'flex items-center justify-center',
          'bg-indigo-600 hover:bg-indigo-500',
          'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-indigo-600',
          'transition-colors duration-200'
        )}
      >
        {isLoading ? (
          <Loader2 className="w-5 h-5 text-white animate-spin" />
        ) : (
          <Send className="w-5 h-5 text-white" />
        )}
      </button>
    </div>
  );
}
