import React, { useState } from 'react';
import { Message } from '../types';

interface ChatMessageProps {
  message: Message;
  sendFeedback?: (feedback: "like" | "dislike", messageId: string) => void;
  feedback?: "like" | "dislike" | null;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ 
  message, 
  sendFeedback,
  feedback
}) => {
  const isUser = message.role === 'user';
  const [hasFeedback, setHasFeedback] = useState(false);
  
  const handleFeedback = (type: "like" | "dislike") => {
    if (!hasFeedback && sendFeedback) {
      sendFeedback(type, message.timestamp);
      setHasFeedback(true);
    }
  };
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} group`}>
      <div
        className={`relative max-w-[85%] p-3 rounded-lg chat-message ${
          isUser
            ? 'bg-indigo-600 text-white rounded-br-none'
            : 'bg-gray-100 text-gray-800 rounded-bl-none'
        }`}
      >
        <p className="text-base whitespace-pre-wrap leading-relaxed">
          {message.content}
        </p>
        <div className="flex items-center justify-between mt-1.5 text-xs">
          <span className={`${isUser ? 'opacity-50' : 'text-gray-500'}`}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          
          {!isUser && sendFeedback && !hasFeedback && (
            <div className="flex items-center gap-2 ml-4 transition-opacity duration-200">
              <button
                onClick={() => handleFeedback('like')}
                className={`p-1.5 rounded-full ${
                  feedback === 'like' 
                    ? 'bg-green-100 text-green-600' 
                    : 'hover:bg-gray-200 text-gray-400 hover:text-gray-600'
                }`}
                disabled={hasFeedback}
                aria-label="Like"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
                  <path d="M8.864.046C7.908-.193 7.02.53 6.956 1.466c-.072 1.051-.23 2.016-.428 2.59-.125.36-.479 1.013-1.04 1.639-.557.623-1.282 1.178-2.131 1.41C2.685 7.288 2 7.87 2 8.72v4.001c0 .845.682 1.464 1.448 1.545 1.07.114 1.564.415 2.068.723l.048.03c.272.165.578.348.97.484.397.136.861.217 1.466.217h3.5c.937 0 1.599-.477 1.934-1.064a1.86 1.86 0 0 0 .254-.912c0-.152-.023-.312-.077-.464.201-.263.38-.578.488-.901.11-.33.172-.762.004-1.149.069-.13.12-.269.159-.403.077-.27.113-.568.113-.857 0-.288-.036-.585-.113-.856a2.144 2.144 0 0 0-.138-.362 1.9 1.9 0 0 0 .234-1.734c-.206-.592-.682-1.1-1.2-1.272-.847-.282-1.803-.276-2.516-.211a9.84 9.84 0 0 0-.443.05 9.365 9.365 0 0 0-.062-4.509A1.38 1.38 0 0 0 9.125.111L8.864.046z"/>
                </svg>
              </button>
              <button
                onClick={() => handleFeedback('dislike')}
                className={`p-1.5 rounded-full ${
                  feedback === 'dislike' 
                    ? 'bg-red-100 text-red-600' 
                    : 'hover:bg-gray-200 text-gray-400 hover:text-gray-600'
                }`}
                disabled={hasFeedback}
                aria-label="Dislike"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 16 16">
                  <path d="M8.864 15.674c-.956.24-1.843-.484-1.908-1.42-.072-1.05-.23-2.015-.428-2.59-.125-.36-.479-1.012-1.04-1.638-.557-.624-1.282-1.179-2.131-1.41C2.685 8.432 2 7.85 2 7V3c0-.845.682-1.464 1.448-1.546 1.07-.113 1.564-.415 2.068-.723l.048-.029c.272-.166.578-.349.97-.484C6.931.08 7.395 0 8 0h3.5c.937 0 1.599.478 1.934 1.064.164.287.254.607.254.913 0 .152-.023.312-.077.464.201.262.38.577.488.9.11.33.172.762.004 1.15.069.13.12.268.159.403.077.27.113.567.113.856 0 .289-.036.586-.113.856-.035.12-.08.244-.138.363.394.571.418 1.2.234 1.733-.206.592-.682 1.1-1.2 1.272-.847.283-1.803.276-2.516.211a9.877 9.877 0 0 1-.443-.05 9.364 9.364 0 0 1-.062 4.51c-.138.508-.55.848-1.012.964l-.261.065z"/>
                </svg>
              </button>
            </div>
          )}
          
          {hasFeedback && (
            <div className="text-xs text-gray-400 mt-1">
              Thanks for your feedback!
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
