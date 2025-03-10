"use client";

import { useState, useEffect, useRef } from "react";
import {
  Message,
  ChatRequest,
  ChatResponse,
  TeachingStrategy,
  UserState,
  LearningMetrics,
} from "../types";
import AdvancedMetrics from "./AdvancedMetrics";
import TeachingStrategyDisplay from "./TeachingStrategyDisplay";
import ChatMessage from "./ChatMessage";
import DetailedMetricsPanel from "./DetailedMetricsPanel";
import TopicKnowledgeGraph from "./TopicKnowledgeGraph";
import { MetricsDisplay } from "./LearningMetrics";
import AdvancedMetricsWrapper from "./AdvancedMetricsWrapper";
import FollowUpSuggestions from "./FollowUpSuggestions";

interface ChatInterfaceProps {
  userId: string;
}

// Update the API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type MetricsTab = "overview" | "detailed" | "topics" | "strategy";

export default function ChatInterface({ userId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentStrategy, setCurrentStrategy] =
    useState<TeachingStrategy | null>(null);
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
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

  const [detailedMetrics, setDetailedMetrics] = useState({
    topicMastery: {},
    learningSpeed: 0,
    interactionQuality: 0,
    contextUtilization: 0,
  });

  const [learningHistory, setLearningHistory] = useState([]);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);
  const [activeTab, setActiveTab] = useState<MetricsTab>("overview");
  const [feedbacks, setFeedbacks] = useState<{
    [key: string]: "like" | "dislike" | null;
  }>({});

  // Add state for follow-up suggestions
  const [followUpSuggestions, setFollowUpSuggestions] = useState<{
    [messageId: string]: string[];
  }>({});

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
        if (response.ok) {
          const userData = await response.json();
          setUserState(userData);

          if (userData.chat_history) {
            setMessages(
              userData.chat_history.map((msg: any) => ({
                role: msg.role,
                content: msg.content,
                timestamp: msg.timestamp,
              }))
            );
          }

          // Set learning history
          if (userData.learning_history) {
            setLearningHistory(userData.learning_history);
          }
        }

        // Also fetch detailed metrics if available
        try {
          const metricsResponse = await fetch(
            `${API_BASE_URL}/learning-progress/${userId}`
          );
          if (metricsResponse.ok) {
            const metricsData = await metricsResponse.json();
            setDetailedMetrics(metricsData);
          }
        } catch (error) {
          console.warn("Could not fetch detailed metrics:", error);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
        setError("Failed to load user data. Please refresh the page.");
      }
    };

    fetchUserState();
  }, [userId]);

  // Helper function to extract a topic from the response
  const extractTopicFromResponse = (response: string): string => {
    const firstSentence = response.split(".")[0];
    const words = firstSentence.split(" ");

    // Try to extract a meaningful phrase (2-3 words)
    for (let i = 0; i < words.length - 1; i++) {
      const word = words[i].toLowerCase();
      if (
        word.length > 4 &&
        ![
          "about",
          "these",
          "those",
          "there",
          "their",
          "would",
          "could",
          "should",
        ].includes(word)
      ) {
        return word.charAt(0).toUpperCase() + word.slice(1);
      }
    }

    return "this topic";
  };

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

      // Generate follow-up suggestions for the last assistant message
      const suggestedQuestions = [
        `Can you explain more about ${extractTopicFromResponse(
          data.response
        )}?`,
        `How does this relate to ${
          userState.recent_topics[0] || "other topics"
        }?`,
        `What's a practical example of this concept?`,
      ];

      setFollowUpSuggestions((prev) => ({
        ...prev,
        [assistantMessage.timestamp]: suggestedQuestions,
      }));

      // Update user state
      const stateResponse = await fetch(`${API_BASE_URL}/user/${userId}`);
      if (stateResponse.ok) {
        const newState = await stateResponse.json();
        setUserState(newState);

        // Update learning history if available
        if (newState.learning_history) {
          setLearningHistory(newState.learning_history);
        }
      }
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to get response. Please try again.");
    } finally {
      setLoading(false);
      setThinking(false);
    }
  };

  const sendFeedback = async (
    feedback: "like" | "dislike",
    messageId: string
  ) => {
    try {
      const res = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          message_id: messageId,
          feedback: feedback,
        }),
      });

      if (res.ok) {
        setFeedbacks((prev) => ({ ...prev, [messageId]: feedback }));
      }
    } catch (error) {
      console.error("Feedback error:", error);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  const renderMetricsContent = () => {
    switch (activeTab) {
      case "overview":
        return <MetricsDisplay metrics={metrics} />;
      case "detailed":
        return (
          <DetailedMetricsPanel
            metrics={detailedMetrics}
            learningHistory={learningHistory}
          />
        );
      case "topics":
        return (
          <TopicKnowledgeGraph
            topics={userState.recent_topics}
            mastery={detailedMetrics?.topicMastery}
          />
        );
      case "strategy":
        return <TeachingStrategyDisplay strategy={currentStrategy} />;
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 min-h-[600px] h-[calc(100vh-12rem)]">
      {/* Main Chat Section */}
      <div className="xl:col-span-8 flex flex-col bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        {/* Chat Header */}
        <div className="px-6 py-4 bg-white border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse"></div>
              <h2 className="text-lg font-semibold text-gray-800">
                Interactive Learning Session
              </h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-500">
                {messages.length} messages
              </span>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 px-6 py-4 overflow-y-auto bg-gray-50 scrollbar-thin">
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-4 border border-red-100 animate-fade-in">
              {error}
            </div>
          )}

          {messages.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center h-full text-center p-6">
              <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center mb-4">
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
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-800 mb-2">
                Start a new conversation
              </h3>
              <p className="text-gray-500 max-w-sm">
                Ask any question and I'll help you learn. I adapt to your
                learning style and interests.
              </p>
            </div>
          )}

          <div className="space-y-6">
            {messages.map((msg, idx) => (
              <div key={idx} className="space-y-2">
                <ChatMessage
                  message={msg}
                  sendFeedback={sendFeedback}
                  feedback={feedbacks[msg.timestamp]}
                />

                {/* Add follow-up suggestions after assistant messages */}
                {msg.role === "assistant" &&
                  followUpSuggestions[msg.timestamp] && (
                    <div className="ml-4">
                      <FollowUpSuggestions
                        suggestions={followUpSuggestions[msg.timestamp]}
                        onSuggestionClick={handleSuggestionClick}
                      />
                    </div>
                  )}
              </div>
            ))}
            {thinking && (
              <div className="flex items-center space-x-2 p-4 bg-gray-50 rounded-lg w-fit animate-pulse">
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-indigo-600 rounded-full animate-bounce delay-200" />
              </div>
            )}
          </div>
          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <div className="px-6 py-4 bg-white border-t border-gray-200">
          <form onSubmit={sendMessage} className="flex gap-3">
            <div className="relative flex-1">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={loading}
                placeholder="Ask your question..."
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent text-gray-800 placeholder-gray-400 transition-all duration-200"
              />
              {loading && (
                <div className="absolute right-4 top-1/2 -translate-y-1/2">
                  <div className="w-4 h-4 border-2 border-violet-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
              )}
            </div>
            <button
              type="submit"
              disabled={loading}
              className={`px-6 py-3 rounded-lg font-medium text-sm transition-all duration-200 flex items-center gap-2 min-w-[100px] justify-center
                ${
                  loading
                    ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-violet-600 to-indigo-600 text-white hover:from-violet-700 hover:to-indigo-700 hover:shadow-md active:scale-95"
                }`}
            >
              {loading ? "Processing..." : "Send"}
            </button>
          </form>
        </div>
      </div>

      {/* Right Sidebar - Analytics */}
      <div className="xl:col-span-4 flex flex-col gap-4 h-full overflow-y-auto">
        {/* Enhanced Tabs */}
        <div className="bg-white rounded-lg p-2 shadow-md border border-gray-200">
          <div className="flex gap-1">
            {[
              { id: "overview", label: "Overview", icon: "📊" },
              { id: "detailed", label: "Analysis", icon: "📈" },
              { id: "topics", label: "Topics", icon: "📚" },
              { id: "strategy", label: "Strategy", icon: "🎯" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as MetricsTab)}
                className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-all duration-200 flex items-center justify-center gap-1.5
                  ${
                    activeTab === tab.id
                      ? "bg-violet-600 text-white shadow-sm"
                      : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                  }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Metrics Content */}
        <div className="flex-1 bg-white rounded-lg shadow-md border border-gray-200 p-4 overflow-y-auto scrollbar-thin">
          {renderMetricsContent()}
        </div>

        {/* User State Summary */}
        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-medium text-gray-700">
              Learning Status
            </h3>
            <span className="text-xs text-indigo-600 font-medium bg-indigo-50 px-2 py-0.5 rounded-full">
              Session #{userState.session_metrics.messages_count}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Knowledge</div>
              <div className="text-sm font-medium">
                {(userState.knowledge_level * 100).toFixed(0)}%
              </div>
              <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{ width: `${userState.knowledge_level * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-gray-50 rounded-md p-2">
              <div className="text-xs text-gray-500">Engagement</div>
              <div className="text-sm font-medium">
                {(userState.engagement * 100).toFixed(0)}%
              </div>
              <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${userState.engagement * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Advanced Metrics Integration */}
        {userState && (
          <AdvancedMetricsWrapper
            userState={userState}
            currentStrategy={currentStrategy}
            learningHistory={learningHistory}
            currentMetrics={metrics}
            detailedMetrics={detailedMetrics}
          />
        )}
      </div>
    </div>
  );
}
