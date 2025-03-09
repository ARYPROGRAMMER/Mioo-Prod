import { Message } from "../types";
import { motion } from "framer-motion";

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isAssistant = message.role === "assistant";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex ${isAssistant ? "justify-start" : "justify-end"}`}
    >
      <div
        className={`
          max-w-[80%] rounded-xl p-4 shadow-sm
          ${
            isAssistant
              ? "bg-gradient-to-r from-gray-50 to-indigo-50 text-gray-800 border border-indigo-100"
              : "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white"
          }
        `}
      >
        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
      </div>
    </motion.div>
  );
}
