"use client";

import { useState } from "react";

interface FollowUpSuggestionsProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

export default function FollowUpSuggestions({
  suggestions,
  onSuggestionClick,
}: FollowUpSuggestionsProps) {
  const [expanded, setExpanded] = useState(true);

  // If no suggestions, don't render anything
  if (!suggestions || suggestions.length === 0) {
    return null;
  }

  return (
    <div className="py-2 px-1">
      <div
        className={`transition-all duration-300 ${
          expanded ? "max-h-40" : "max-h-8 overflow-hidden"
        }`}
      >
        <div
          className="flex items-center justify-between text-xs text-indigo-600 cursor-pointer hover:text-indigo-800 mb-2"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="font-medium">Follow-up questions</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform ${
              expanded ? "rotate-180" : "rotate-0"
            }`}
          >
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        </div>

        <div className="space-y-2">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              onClick={() => onSuggestionClick(suggestion)}
              className="w-full text-left px-3 py-1.5 rounded-md bg-white border border-indigo-100 hover:border-indigo-300 hover:bg-indigo-50 text-xs text-gray-700 transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
