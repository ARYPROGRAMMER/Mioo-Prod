"use client";

import { TeachingStrategy } from "../types";

interface TeachingStrategyDisplayProps {
  strategy: TeachingStrategy;
}

export default function TeachingStrategyDisplay({
  strategy,
}: TeachingStrategyDisplayProps) {
  const getStyleIcon = (style: string) => {
    switch (style) {
      case "detailed":
        return "📚";
      case "concise":
        return "📝";
      case "interactive":
        return "🤝";
      case "analogy-based":
        return "🔄";
      case "step-by-step":
        return "📋";
      default:
        return "📖";
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case "high":
        return "text-red-500 dark:text-red-400";
      case "medium":
        return "text-yellow-500 dark:text-yellow-400";
      case "low":
        return "text-green-500 dark:text-green-400";
      default:
        return "text-blue-500 dark:text-blue-400";
    }
  };

  const getExamplesIcon = (examples: string) => {
    switch (examples) {
      case "many":
        return "⭐⭐⭐";
      case "some":
        return "⭐⭐";
      case "few":
        return "⭐";
      default:
        return "⭐⭐";
    }
  };

  return (
    <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200">
        Current Teaching Approach
      </h3>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="text-2xl mb-2">{getStyleIcon(strategy.style)}</div>
          <div className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Style
          </div>
          <div className="text-base font-semibold capitalize">
            {strategy.style.replace("-", " ")}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div
            className={`text-lg font-bold ${getComplexityColor(
              strategy.complexity
            )}`}
          >
            {strategy.complexity.toUpperCase()}
          </div>
          <div className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Complexity
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-sm">
          <div className="text-xl mb-1">
            {getExamplesIcon(strategy.examples)}
          </div>
          <div className="text-sm font-medium text-gray-600 dark:text-gray-400">
            Examples
          </div>
          <div className="text-base font-semibold capitalize">
            {strategy.examples}
          </div>
        </div>
      </div>
    </div>
  );
}
