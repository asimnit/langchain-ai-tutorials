import { useState } from 'react';
import { ChevronDown, ChevronRight, Wrench, CheckCircle2, Loader2 } from 'lucide-react';
import type { ToolExecution } from '../types';

interface ToolOutputProps {
  execution: ToolExecution;
}

function formatToolName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(', ');
}

function formatResult(result: string): string {
  try {
    const parsed = JSON.parse(result);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return result;
  }
}

export function ToolOutput({ execution }: ToolOutputProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isCompleted = execution.status === 'completed';

  return (
    <div className="tool-card my-2 rounded-lg border border-gray-700 bg-gray-800/50 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-700/50 transition-colors"
      >
        {/* Expand/Collapse icon */}
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400" />
        )}

        {/* Tool icon */}
        <Wrench className="w-4 h-4 text-indigo-400" />

        {/* Tool name and args */}
        <div className="flex-1 text-left">
          <span className="text-sm font-medium text-indigo-300">
            {formatToolName(execution.name)}
          </span>
          <span className="text-xs text-gray-500 ml-2">
            ({formatArgs(execution.args)})
          </span>
        </div>

        {/* Status indicator */}
        {isCompleted ? (
          <CheckCircle2 className="w-4 h-4 text-green-400" />
        ) : (
          <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
        )}
      </button>

      {/* Expandable content */}
      {isExpanded && execution.result && (
        <div className="px-3 pb-3 border-t border-gray-700">
          <div className="mt-2">
            <div className="text-xs text-gray-500 mb-1">Output:</div>
            <pre className="json-content bg-gray-900 rounded p-2 text-green-300 overflow-x-auto max-h-60 overflow-y-auto">
              {formatResult(execution.result)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

interface ToolExecutionListProps {
  executions: ToolExecution[];
}

export function ToolExecutionList({ executions }: ToolExecutionListProps) {
  if (executions.length === 0) return null;

  return (
    <div className="space-y-1">
      {executions.map((execution) => (
        <ToolOutput key={execution.id} execution={execution} />
      ))}
    </div>
  );
}
