"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Message,
  ChatResponse,
  TeachingStrategy,
  UserState,
  LearningMetrics,
  FeedbackRequest,
} from "../types";
import Chat from "./Chat";
import AdvancedMetricsWrapper from "./AdvancedMetricsWrapper";
import TeachingStrategyDisplay from "./TeachingStrategyDisplay";
import DetailedMetricsPanel from "./DetailedMetricsPanel";
import TopicKnowledgeGraph from "./TopicKnowledgeGraph";
import { MetricsDisplay } from "./LearningMetrics";
import FollowUpSuggestions from "./FollowUpSuggestions";
import ChatMessage from "./ChatMessage";

interface ChatInterfaceProps {
  userId: string;
}

// Update the API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type MetricsTab = "overview" | "detailed" | "topics" | "strategy";

export default function ChatInterface({ userId }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
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
  const [error, setError] = useState<string | null>(null);
  const [detailedMetrics, setDetailedMetrics] = useState({
    topicMastery: {} as Record<string, number>,
    learningSpeed: 0.5,
    interactionQuality: 0.5,
    contextUtilization: 0.5,
  });
  const [learningHistory, setLearningHistory] = useState([]);
  const [activeTab, setActiveTab] = useState<MetricsTab>("overview");
  const [feedbacks, setFeedbacks] = useState<{
    [key: string]: "like" | "dislike" | null;
  }>({});
  const [followUpSuggestions, setFollowUpSuggestions] = useState<{
    [messageId: string]: string[];
  }>({});

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
        const metricsResponse = await fetch(
          `${API_BASE_URL}/learning-progress/${userId}`
        );
        if (metricsResponse.ok) {
          const metricsData = await metricsResponse.json();
          setDetailedMetrics(metricsData);
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
        setError("Failed to load user data. Please refresh the page.");
      }
    };

    fetchUserState();
  }, [userId]);

  // Optimize initial greeting
  useEffect(() => {
    const initializeChat = async () => {
      if (messages.length === 0) {
        // Start with minimal greeting
        const initialMessage: Message = {
          role: "assistant",
          content: "Hi! I'm your AI tutor.",
          timestamp: new Date().toISOString()
        };
        setMessages([initialMessage]);
        
        // Fetch full personalized greeting in background
        try {
          const response = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: "__init__",
              user_id: userId,
              is_initial: true
            })
          });

          if (response.ok) {
            const data = await response.json();
            // Replace initial greeting with personalized one
            setMessages([{
              role: "assistant",
              content: data.response,
              timestamp: new Date().toISOString()
            }]);
          }
        } catch (error) {
          console.error("Error fetching full greeting:", error);
        }
      }
    };

    initializeChat();
  }, [userId, messages.length]);

  // Helper function to extract a topic from the response
  const extractTopicFromResponse = (response: string): string => {
    const firstSentence = response.split(".")[0];
    const words = firstSentence.split(" ");

    // Try to extract a meaningful phrase
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

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || loading) return;

    const userMessage: Message = {
      role: "user",
      content: content.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setError(null);

    // Clear input and disable it
    if (inputRef.current) {
      inputRef.current.value = '';
      inputRef.current.disabled = true;
    }

    try {
      // Store previous message to check for context changes
      const prevMessages = messages;
      const lastBotMessage = messages.length > 0 ? 
        messages.filter(m => m.role === "assistant").pop() : null;

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          user_id: userId,
          context: {
            isShortResponse: content.trim().split(/\s+/).length <= 3,
            lastMessageTimestamp: lastBotMessage?.timestamp
          }
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

      // Generate follow-up suggestions
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
        if (newState.learning_history) {
          setLearningHistory(newState.learning_history);
        }
      }

      // If this is a short response like "no", scroll to make sure the response is visible
      if (content.trim().split(/\s+/).length <= 3) {
        setTimeout(() => {
          const messagesContainer = document.querySelector(".overflow-y-auto");
          if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
          }
        }, 100);
      }
    } catch (error) {
      console.error("Error:", error);
      setError("Failed to get response. Please try again.");
    } finally {
      setLoading(false);
      // Re-enable input
      if (inputRef.current) {
        inputRef.current.disabled = false;
      }
    }
  };

  const sendFeedback = async (
    feedback: "like" | "dislike",
    messageId: string
  ) => {
    try {
      console.log(`Sending feedback: ${feedback} for message: ${messageId}`);
      
      const feedbackRequest: FeedbackRequest = {
        user_id: userId,
        message_id: messageId,
        feedback: feedback,
      };
      
      // Set feedback state immediately for better UX
      setFeedbacks((prev) => ({ ...prev, [messageId]: feedback }));
      
      const res = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedbackRequest),
      });

      if (res.ok) {
        // Update user state to reflect feedback
        const stateResponse = await fetch(`${API_BASE_URL}/user/${userId}`);
        if (stateResponse.ok) {
          const newState = await stateResponse.json();
          setUserState(newState);
        }
      } else {
        // If server returns error, revert the feedback
        setFeedbacks((prev) => ({ ...prev, [messageId]: null }));
        throw new Error(`Failed to send feedback: ${res.statusText}`);
      }
    } catch (error) {
      console.error("Feedback error:", error);
      setError("Failed to send feedback. Please try again.");
    }
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

  // Show follow-up suggestions after the last assistant message
  const lastAssistantMessage = messages
    .slice()
    .reverse()
    .find((msg) => msg.role === "assistant");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget as HTMLFormElement);
    const message = formData.get('message')?.toString() || '';
    
    if (message.trim() && !loading) {
      // Disable input immediately
      if (inputRef.current) {
        inputRef.current.disabled = true;
      }
      
      await handleSendMessage(message);
      
      // Clear the input
      if (inputRef.current) {
        inputRef.current.value = '';
        inputRef.current.disabled = false;
      }
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 xl:grid-cols-4 gap-4 h-full">
      {/* Main Chat Section */}
      <div className="lg:col-span-2 xl:col-span-3 h-full flex flex-col">
        <div className="flex flex-col h-full bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
          {/* Chat Header */}
          <div className="px-6 py-3 bg-white border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <h3 className="text-base font-medium text-gray-700">Learning Conversation</h3>
              </div>
              <div className="text-sm text-gray-500">
                {userState.recent_topics.length > 0 && (
                  <span>Topics: {userState.recent_topics.slice(0, 3).join(", ")}</span>
                )}
              </div>
            </div>
          </div>
          
          {/* Messages Container - Fix scrolling issue */}
          <div className="flex-1 p-4 space-y-6 overflow-y-auto bg-gray-50 scrollbar-thin min-h-[300px] max-h-[calc(100vh-300px)]">
            {/* ...existing code for empty state... */}
            
            {messages.length === 0 ? (
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
                <h3 className="text-lg font-medium text-gray-800 mb-2">Start a conversation</h3>
                <p className="text-gray-500 max-w-sm text-base">
                  Ask any question and I'll adapt to your learning style.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((msg, idx) => (
                  <ChatMessage
                    key={idx}
                    message={msg}
                    sendFeedback={msg.role === "assistant" ? sendFeedback : undefined}
                    feedback={feedbacks[msg.timestamp]}
                  />
                ))}
              </div>
            )}
            
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 rounded-lg rounded-bl-none p-3">
                  <div className="flex space-x-2">
                    <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce" 
                         style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-3 h-3 bg-gray-400 rounded-full animate-bounce" 
                         style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Form - Fixed Enter key issue with onKeyDown */}
          <form onSubmit={handleSubmit} className="px-4 py-3 bg-white border-t border-gray-200">
            <div className="flex items-end space-x-2">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  name="message"
                  placeholder="Type your message..."
                  className="w-full resize-none rounded-lg border border-gray-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 p-3 pr-10 text-base"
                  rows={2}
                  style={{ minHeight: '60px', maxHeight: '120px' }}
                  disabled={loading}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      const message = e.currentTarget.value;
                      if (message.trim() && !loading) {
                        handleSendMessage(message);
                        e.currentTarget.value = '';
                      }
                    }
                  }}
                ></textarea>
                {loading && (
                  <div className="absolute right-3 bottom-3">
                    <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>
              <button
                type="submit"
                disabled={loading}
                className="px-5 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex-shrink-0 font-medium text-base"
              >
                Send
              </button>
            </div>
          </form>
        </div>
        
        {/* Follow-up suggestions */}
        {lastAssistantMessage && followUpSuggestions[lastAssistantMessage.timestamp] && !loading && (
          <div className="mt-3">
            <FollowUpSuggestions
              suggestions={followUpSuggestions[lastAssistantMessage.timestamp]}
              onSuggestionClick={(suggestion) => handleSendMessage(suggestion)}
            />
          </div>
        )}
      </div>

      {/* Metrics Section */}
      <div className="lg:col-span-1 h-full flex flex-col gap-3">
        {/* Metrics Tabs */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="grid grid-cols-4 gap-1 p-1">
            {[
              { id: "overview", label: "Overview", icon: "📊" },
              { id: "detailed", label: "Analysis", icon: "📈" },
              { id: "topics", label: "Topics", icon: "📚" },
              { id: "strategy", label: "Strategy", icon: "🎯" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as MetricsTab)}
                className={`py-2 px-1 rounded text-xs font-medium transition-all flex flex-col items-center justify-center ${
                  activeTab === tab.id
                    ? "bg-indigo-600 text-white"
                    : "text-gray-600 hover:bg-gray-50"
                }`}
              >
                <span className="text-base mb-1">{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Metrics Content */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 flex-1 overflow-y-auto scrollbar-thin">
          {error ? (
            <div className="text-red-500 text-sm p-3 bg-red-50 rounded-md">
              {error}
            </div>
          ) : (
            renderMetricsContent()
          )}
        </div>

        {/* Learning Status Summary */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-xs font-medium text-gray-700">Learning Progress</h3>
            <span className="text-xs text-indigo-600 font-medium bg-indigo-50 px-2 py-0.5 rounded-full">
              {userState.session_metrics.topics_covered.length} topics
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div className="bg-gray-50 rounded p-2">
              <div className="text-xs text-gray-500">Knowledge</div>
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">
                  {(userState.knowledge_level * 100).toFixed(0)}%
                </div>
                {metrics?.knowledge_gain && metrics.knowledge_gain > 0 && (
                  <span className="text-xs text-green-600 bg-green-50 px-1 py-0.5 rounded">
                    +{(metrics.knowledge_gain * 100).toFixed(0)}%
                  </span>
                )}
              </div>
              <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full"
                  style={{ width: `${userState.knowledge_level * 100}%` }}
                />
              </div>
            </div>

            <div className="bg-gray-50 rounded p-2">
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
      </div>
    </div>
  );
}
