"use client";

import { TeachingStrategy } from "../types";

interface TeachingStrategyDisplayProps {
  strategy: TeachingStrategy | null;
}

export default function TeachingStrategyDisplay({
  strategy,
}: TeachingStrategyDisplayProps) {
  if (!strategy) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-center p-6">
        <div>
          <div className="mb-3 mx-auto w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="text-gray-400"
            >
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
              <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
            </svg>
          </div>
          <p className="text-sm mb-1">No teaching strategy selected yet</p>
          <p className="text-xs">
            Start a conversation to see the AI's approach
          </p>
        </div>
      </div>
    );
  }

  const getStyleIcon = (style: string) => {
    switch (style?.toLowerCase()) {
      case "detailed":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-indigo-500"
          >
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
          </svg>
        );
      case "concise":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-blue-500"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        );
      case "interactive":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-green-500"
          >
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
          </svg>
        );
      case "analogy-based":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-purple-500"
          >
            <path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path>
          </svg>
        );
      case "step-by-step":
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-yellow-500"
          >
            <line x1="8" y1="6" x2="21" y2="6"></line>
            <line x1="8" y1="12" x2="21" y2="12"></line>
            <line x1="8" y1="18" x2="21" y2="18"></line>
            <line x1="3" y1="6" x2="3.01" y2="6"></line>
            <line x1="3" y1="12" x2="3.01" y2="12"></line>
            <line x1="3" y1="18" x2="3.01" y2="18"></line>
          </svg>
        );
      default:
        return (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-gray-500"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
          </svg>
        );
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity?.toLowerCase()) {
      case "high":
        return "text-red-600 bg-red-50";
      case "medium":
        return "text-yellow-600 bg-yellow-50";
      case "low":
        return "text-green-600 bg-green-50";
      default:
        return "text-blue-600 bg-blue-50";
    }
  };

  const getExamplesBadge = (examples: string) => {
    switch (examples?.toLowerCase()) {
      case "many":
        return { color: "bg-purple-500", count: 3 };
      case "some":
        return { color: "bg-blue-500", count: 2 };
      case "few":
        return { color: "bg-gray-500", count: 1 };
      default:
        return { color: "bg-blue-500", count: 2 };
    }
  };

  const examplesBadge = getExamplesBadge(strategy.examples);

  return (
    <div className="space-y-6 h-full">
      <div className="border-b border-gray-200 pb-3">
        <h3 className="text-base font-semibold text-gray-800">
          Teaching Strategy
        </h3>
        <p className="text-xs text-gray-500 mt-1">
          The AI adapts its approach based on your learning style
        </p>
      </div>

      <div className="grid gap-4">
        {/* Style */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-indigo-50 rounded-lg">
              {getStyleIcon(strategy.style)}
            </div>
            <div>
              <div className="font-medium text-gray-900 mb-1">
                {strategy.style?.replace("-", " ") || "Default"}
              </div>
              <div className="text-sm text-gray-500">
                {strategy.style === "detailed" &&
                  "In-depth explanations with thorough coverage of concepts"}
                {strategy.style === "concise" &&
                  "Brief, clear explanations focused on key points"}
                {strategy.style === "interactive" &&
                  "Engaging format with questions and exercises"}
                {strategy.style === "analogy-based" &&
                  "Using metaphors and comparisons for better understanding"}
                {strategy.style === "step-by-step" &&
                  "Sequential explanations that build progressively"}
                {!strategy.style && "Balanced teaching approach"}
              </div>
            </div>
          </div>
        </div>

        {/* Complexity & Examples */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
            <div className="text-sm text-gray-500 mb-1">Complexity</div>
            <div
              className={`inline-block px-2 py-1 rounded text-xs font-medium ${getComplexityColor(
                strategy.complexity
              )}`}
            >
              {strategy.complexity?.toUpperCase() || "ADAPTIVE"}
            </div>
            <div className="mt-2 text-xs text-gray-500">
              {strategy.complexity === "high" &&
                "Advanced concepts with technical depth"}
              {strategy.complexity === "medium" &&
                "Balanced approach for intermediate understanding"}
              {strategy.complexity === "low" &&
                "Simplified explanations for building fundamentals"}
              {strategy.complexity === "adjustable" &&
                "Adapts based on topic complexity"}
              {!strategy.complexity && "Adjusts dynamically to your progress"}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-4">
            <div className="text-sm text-gray-500 mb-1">Examples</div>
            <div className="flex items-center gap-1 mb-2">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className={`w-2 h-2 rounded-full ${
                    i < examplesBadge.count
                      ? examplesBadge.color
                      : "bg-gray-200"
                  }`}
                />
              ))}
              <span className="text-xs text-gray-700 ml-1">
                {strategy.examples || "Default"}
              </span>
            </div>
            <div className="text-xs text-gray-500">
              {strategy.examples === "many" &&
                "Multiple detailed examples for thorough understanding"}
              {strategy.examples === "some" &&
                "Selected examples to illustrate key points"}
              {strategy.examples === "few" &&
                "Minimal, focused examples for clarity"}
              {!strategy.examples && "Balanced number of relevant examples"}
            </div>
          </div>
        </div>

        {/* Learning tips */}
        <div className="bg-indigo-50 rounded-lg p-4">
          <h4 className="text-sm font-medium text-indigo-900 mb-2">
            Learning Tip
          </h4>
          <p className="text-xs text-indigo-800">
            {strategy.style === "detailed" &&
              "Take notes and focus on connecting concepts for better retention."}
            {strategy.style === "concise" &&
              "Create your own examples to test your understanding of key points."}
            {strategy.style === "interactive" &&
              "Actively participate by answering questions to maximize learning."}
            {strategy.style === "analogy-based" &&
              "Try creating your own analogies to reinforce understanding."}
            {strategy.style === "step-by-step" &&
              "Follow each step carefully before moving to the next one."}
            {!strategy.style &&
              "Ask follow-up questions to deepen your understanding."}
          </p>
        </div>
      </div>
    </div>
  );
}
