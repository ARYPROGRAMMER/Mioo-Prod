"use client";

import React from 'react';

interface FollowUpSuggestionsProps {
  suggestions: string[];
  onSuggestionClick: (suggestion: string) => void;
}

const FollowUpSuggestions: React.FC<FollowUpSuggestionsProps> = ({
  suggestions,
  onSuggestionClick
}) => {
  return (
    <div className="mt-2">
      <p className="text-sm text-gray-500 mb-1.5">Follow-up questions:</p>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion)}
            className="text-sm py-1.5 px-3 bg-white border border-gray-200 rounded-full text-indigo-600 hover:bg-indigo-50 hover:border-indigo-200 transition-colors duration-200 shadow-sm"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default FollowUpSuggestions;
