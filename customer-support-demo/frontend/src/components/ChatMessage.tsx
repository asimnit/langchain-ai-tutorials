import { User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage as ChatMessageType } from '../types';
import { ToolExecutionList } from './ToolOutput';
import { ThinkingIndicator } from './ThinkingIndicator';
import clsx from 'clsx';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={clsx(
        'message-fade-in flex gap-3 py-4',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-indigo-600' : 'bg-gray-700'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-indigo-300" />
        )}
      </div>

      {/* Message content */}
      <div
        className={clsx(
          'flex-1 max-w-[80%]',
          isUser ? 'text-right' : 'text-left'
        )}
      >
        {/* Role label */}
        <div className="text-xs text-gray-500 mb-1">
          {isUser ? 'You' : 'Support Agent'}
        </div>

        {/* Thinking indicator */}
        {message.thinking && <ThinkingIndicator message={message.thinking} />}

        {/* Tool executions */}
        {message.toolExecutions && message.toolExecutions.length > 0 && (
          <ToolExecutionList executions={message.toolExecutions} />
        )}

        {/* Message text */}
        {message.content?.trim() && (
          <div
            className={clsx(
              'inline-block rounded-2xl px-4 py-2 max-w-full',
              isUser
                ? 'bg-indigo-600 text-white rounded-tr-sm'
                : 'bg-gray-800 text-gray-100 rounded-tl-sm',
              message.isStreaming && 'typing-cursor'
            )}
          >
            <div className="break-words prose prose-invert prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => <h1 className="text-2xl font-bold mt-3 mb-1">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-xl font-bold mt-3 mb-1">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-lg font-bold mt-3 mb-1">{children}</h3>,
                  hr: () => <hr className="my-3 border-gray-600" />,
                  ul: ({ children }) => <ul className="list-disc ml-4 my-2">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal ml-4 my-2">{children}</ol>,
                  li: ({ children }) => <li className="my-1">{children}</li>,
                  p: ({ children }) => <p className="my-2">{children}</p>,
                  code: ({ className, children }) => {
                    const isBlock = className?.includes('language-');
                    return isBlock ? (
                      <pre className="bg-gray-900 rounded p-2 my-2 overflow-x-auto text-sm">
                        <code>{children}</code>
                      </pre>
                    ) : (
                      <code className="bg-gray-900 px-1 rounded text-indigo-300">{children}</code>
                    );
                  },
                  pre: ({ children }) => <>{children}</>,
                  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                  a: ({ href, children }) => (
                    <a href={href} className="text-indigo-400 hover:underline" target="_blank" rel="noopener noreferrer">
                      {children}
                    </a>
                  ),
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-3">
                      <table className="min-w-full border-collapse border border-gray-600">{children}</table>
                    </div>
                  ),
                  thead: ({ children }) => <thead className="bg-gray-700">{children}</thead>,
                  tbody: ({ children }) => <tbody>{children}</tbody>,
                  tr: ({ children }) => <tr className="border-b border-gray-600">{children}</tr>,
                  th: ({ children }) => <th className="px-3 py-2 text-left font-semibold border border-gray-600">{children}</th>,
                  td: ({ children }) => <td className="px-3 py-2 border border-gray-600">{children}</td>,
                }}
              >
                {message.content.trim()}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Timestamp */}
        <div className="text-xs text-gray-600 mt-1">
          {message.timestamp.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </div>
      </div>
    </div>
  );
}
