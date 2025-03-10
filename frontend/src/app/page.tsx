"use client";
import { useState } from "react";
import { Chat } from "./components/Chat";
import { Message, ChatResponse, UserState, TeachingStrategy } from "./types";
import AdvancedMetricsWrapper  from "./components/AdvancedMetricsWrapper";

interface ExtendedMessage extends Message {
  teachingStrategy?: TeachingStrategy;
}

export default function Home() {
  const [messages, setMessages] = useState<ExtendedMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [userState, setUserState] = useState<UserState | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    setIsLoading(true);
    setError(null);

    // Add user message immediately
    const userMessage: ExtendedMessage = {
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          user_id: userState?.user_id || "new_user",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();

      // Add assistant message with teaching strategy
      const assistantMessage: ExtendedMessage = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toISOString(),
        teachingStrategy: data.teaching_strategy,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Update user state if available
      if ("user_state" in data) {
        setUserState(data.user_state as UserState);
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "An unknown error occurred";
      setError(errorMessage);
      console.error("Error:", error);

      // Add error message to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I apologize, but I encountered an error. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container mx-auto p-4 md:p-8 min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Chat section */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h1 className="text-3xl font-bold text-gray-800">Mioo AI Tutor</h1>
            {error && (
              <div className="text-red-500 text-sm animate-fade-in">
                {error}
              </div>
            )}
          </div>

          <div className="glass-morphism">
            <Chat
              messages={messages}
              isLoading={isLoading}
              onSendMessage={handleSendMessage}
            />
          </div>
        </div>

        {/* Metrics section */}
        <div className="lg:col-span-1">
          {userState && (
            <div className="glass-morphism p-4">
              <AdvancedMetricsWrapper
                userState={userState}
                currentStrategy={
                  messages.length > 0
                    ? messages[messages.length - 1].teachingStrategy ?? null
                    : null
                }
                learningHistory={userState.learning_history || []}
                currentMetrics={userState.current_metrics}
                detailedMetrics={userState.detailed_metrics}
              />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
