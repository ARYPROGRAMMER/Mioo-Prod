"use client";

import { useState, useEffect, useRef } from "react";
import {
  Message,
  ChatResponse,
  TeachingStrategy,
  UserState,
  LearningMetrics,
} from "../types";
import ChatMessage from "./ChatMessage";
import TopicKnowledgeGraph from "./TopicKnowledgeGraph";
import { MetricsDisplay } from "./LearningMetrics";

interface ChatInterfaceProps {
  userId: string;
}

// Update the API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatInterface({ userId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentStrategy, setCurrentStrategy] =
    useState<TeachingStrategy | null>(null);
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [userState, setUserState] = useState<UserState>({
    user_id: userId,
    knowledge_level: 0.5,
    engagement: 0.5,
    interests: [],
    recent_topics: [],
    performance: 0.5,
    chat_history: [],
    last_updated: new Date().toISOString(),
    learning_history: [],
    session_metrics: {
      messages_count: 0,
      average_response_time: 0,
      topics_covered: [],
      learning_rate: 0,
      engagement_trend: [],
    },
  });
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch initial user state
  useEffect(() => {
    const fetchUserState = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/user/${userId}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setUserState(data);
        if (data.chat_history) {
          setMessages(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            data.chat_history.map((msg: any) => ({
              role: msg.role,
              content: msg.content,
              timestamp: msg.timestamp,
            }))
          );
        }
      } catch (error) {
        console.error("Error fetching user state:", error);
        setError("Failed to load user data");
      }
    };
    fetchUserState();
  }, [userId]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setThinking(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setCurrentStrategy(data.teaching_strategy);
      setMetrics(data.metrics);

      const assistantMessage: Message = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Update user state
      const stateResponse = await fetch(`${API_BASE_URL}/user/${userId}`);
      if (stateResponse.ok) {
        const newState = await stateResponse.json();
        setUserState(newState);
      }
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to get response. Please try again.");
    } finally {
      setLoading(false);
      setThinking(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[85vh]">
      {/* Chat Section */}
      <div className="lg:col-span-2 flex flex-col bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
        <div className="flex-1 p-6 overflow-y-auto scrollbar-thin">
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-xl mb-4 border border-red-100">
              {error}
            </div>
          )}
          <div className="space-y-6">
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} />
            ))}
            {thinking && (
              <div className="flex items-center space-x-2 p-4 bg-gray-50 rounded-xl w-fit">
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce delay-200" />
              </div>
            )}
          </div>
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <div className="p-4 border-t border-gray-100 bg-gray-50">
          <form onSubmit={sendMessage} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              placeholder="Ask me anything..."
              className="flex-1 p-4 bg-white border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-gray-800 placeholder-gray-400"
            />
            <button
              type="submit"
              disabled={loading}
              className={`px-6 rounded-xl font-semibold transition-all duration-200
                ${
                  loading
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-indigo-600 to-indigo-700 text-white hover:from-indigo-700 hover:to-indigo-800 hover:shadow-lg"
                }`}
            >
              {loading ? "Thinking..." : "Send"}
            </button>
          </form>
        </div>
      </div>

      {/* Progress Section */}
      <div className="space-y-6">
        {/* Learning Progress */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Learning Progress
          </h2>
          <TopicKnowledgeGraph userId={userId} />
        </div>

        {/* Metrics */}
        <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            Session Metrics
          </h2>
          <MetricsDisplay metrics={metrics} />
        </div>

        {/* Teaching Strategy */}
        {currentStrategy && (
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-gray-100 p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Current Strategy
            </h2>
            <div className="space-y-2">
              <div className="bg-indigo-50/80 backdrop-blur-sm rounded-lg p-4 text-sm text-gray-800">
                <p className="flex justify-between py-1">
                  <span className="font-medium">Style:</span>
                  <span className="text-indigo-700">
                    {currentStrategy.style}
                  </span>
                </p>
                <p className="flex justify-between py-1">
                  <span className="font-medium">Complexity:</span>
                  <span className="text-indigo-700">
                    {currentStrategy.complexity}
                  </span>
                </p>
                <p className="flex justify-between py-1">
                  <span className="font-medium">Examples:</span>
                  <span className="text-indigo-700">
                    {currentStrategy.examples}
                  </span>
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
