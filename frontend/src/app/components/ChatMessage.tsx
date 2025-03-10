import { Message } from "../types";
import { motion } from "framer-motion";

interface ChatMessageProps {
  message: Message;
  sendFeedback?: (feedback: "like" | "dislike", messageId: string) => void;
  feedback?: "like" | "dislike" | null;
}

export default function ChatMessage({
  message,
  sendFeedback,
  feedback,
}: ChatMessageProps) {
  const isAssistant = message.role === "assistant";

  // Format timestamp for display
  const formattedTime = (() => {
    try {
      const date = new Date(message.timestamp);
      return date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch (e) {
      return "";
    }
  })();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isAssistant ? "justify-start" : "justify-end"} mb-3`}
    >
      <div
        className={`
          relative max-w-[85%] sm:max-w-[80%] rounded-2xl px-4 py-3
          ${
            isAssistant
              ? "bg-white text-gray-800 border border-gray-100 shadow-sm"
              : "bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-sm"
          }
        `}
      >
        {/* Message content */}
        <div className="whitespace-pre-wrap leading-relaxed text-sm">
          {message.content}
        </div>

        {/* Timestamp and feedback */}
        <div
          className={`flex items-center mt-2 ${
            isAssistant ? "justify-between" : "justify-end"
          }`}
        >
          {/* Only show feedback buttons for assistant messages */}
          {isAssistant && sendFeedback && (
            <div className="flex space-x-2 mr-3">
              <button
                onClick={() => sendFeedback("like", message.timestamp)}
                disabled={!!feedback}
                className={`p-1.5 rounded-full transition-colors ${
                  feedback === "like"
                    ? "bg-green-100 text-green-700"
                    : "hover:bg-gray-100 text-gray-400 hover:text-green-600"
                }`}
                aria-label="Like this response"
              >
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
                >
                  <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>
                </svg>
              </button>
              <button
                onClick={() => sendFeedback("dislike", message.timestamp)}
                disabled={!!feedback}
                className={`p-1.5 rounded-full transition-colors ${
                  feedback === "dislike"
                    ? "bg-red-100 text-red-700"
                    : "hover:bg-gray-100 text-gray-400 hover:text-red-600"
                }`}
                aria-label="Dislike this response"
              >
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
                >
                  <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"></path>
                </svg>
              </button>
            </div>
          )}

          {/* Timestamp */}
          <div
            className={`text-xs ${
              isAssistant ? "text-gray-400" : "text-white/70"
            }`}
          >
            {formattedTime}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
