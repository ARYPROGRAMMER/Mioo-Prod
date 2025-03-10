"use client";

import { UserState } from "../types";
import type { LearningMetrics } from "../types";

interface LearningMetricsProps {
  userState: UserState;
}

export default function LearningMetrics({ userState }: LearningMetricsProps) {
  const getProgressColor = (value: number) => {
    if (value >= 0.7) return "bg-green-500";
    if (value >= 0.4) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Learning Progress</h3>

      <div className="space-y-4">
        {/* Knowledge Level */}
        <div>
          <div className="flex justify-between mb-1">
            <span>Knowledge Level</span>
            <span>{Math.round(userState.knowledge_level * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${getProgressColor(
                userState.knowledge_level
              )}`}
              style={{ width: `${userState.knowledge_level * 100}%` }}
            />
          </div>
        </div>

        {/* Engagement */}
        <div>
          <div className="flex justify-between mb-1">
            <span>Engagement</span>
            <span>{Math.round(userState.engagement * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${getProgressColor(
                userState.engagement
              )}`}
              style={{ width: `${userState.engagement * 100}%` }}
            />
          </div>
        </div>

        {/* Performance */}
        <div>
          <div className="flex justify-between mb-1">
            <span>Performance</span>
            <span>{Math.round(userState.performance * 100)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${getProgressColor(
                userState.performance
              )}`}
              style={{ width: `${userState.performance * 100}%` }}
            />
          </div>
        </div>

        {/* Interests & Topics */}
        <div className="mt-4">
          <h4 className="font-medium mb-2">Current Interests</h4>
          <div className="flex flex-wrap gap-2">
            {userState.interests.map((interest, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-100 dark:bg-blue-900 rounded-full text-sm"
              >
                {interest}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-4">
          <h4 className="font-medium mb-2">Recent Topics</h4>
          <div className="flex flex-wrap gap-2">
            {userState.recent_topics.map((topic, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-purple-100 dark:bg-purple-900 rounded-full text-sm"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export interface MetricsDisplayProps {
  metrics: LearningMetrics | null;
}

export function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  if (!metrics) {
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
              <path d="M12 20v-6M6 20V10M18 20V4"></path>
            </svg>
          </div>
          <p className="text-sm mb-1">No metrics available yet</p>
          <p className="text-xs">
            Start a conversation to see learning metrics
          </p>
        </div>
      </div>
    );
  }

  const MetricItem = ({
    label,
    value,
    color,
  }: {
    label: string;
    value: number;
    color?: string;
  }) => (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-700">{label}</span>
        <span className="text-sm font-medium text-gray-900">
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            color || "bg-gradient-to-r from-indigo-500 to-violet-500"
          }`}
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
    </div>
  );

  const getMetricColor = (value: number): string => {
    if (value > 0.8) return "bg-green-500";
    if (value > 0.6) return "bg-blue-500";
    if (value > 0.4) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-4">
        <h3 className="text-base font-medium text-gray-900">
          Learning Progress
        </h3>
        <p className="text-xs text-gray-500 mt-1">Current session metrics</p>
      </div>

      <div className="space-y-4 flex-1">
        <MetricItem
          label="Knowledge Gain"
          value={metrics.knowledge_gain}
          color={getMetricColor(metrics.knowledge_gain)}
        />
        <MetricItem
          label="Engagement"
          value={metrics.engagement_level}
          color={getMetricColor(metrics.engagement_level)}
        />
        <MetricItem
          label="Performance"
          value={metrics.performance_score}
          color={getMetricColor(metrics.performance_score)}
        />
        <MetricItem
          label="Strategy Effectiveness"
          value={metrics.strategy_effectiveness}
          color={getMetricColor(metrics.strategy_effectiveness)}
        />
        <MetricItem
          label="Interaction Quality"
          value={metrics.interaction_quality}
          color={getMetricColor(metrics.interaction_quality)}
        />
      </div>

      {/* Tip for better learning */}
      <div className="mt-4 bg-blue-50 p-3 rounded-lg border border-blue-100">
        <div className="flex gap-2">
          <div className="text-blue-500">
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
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
          </div>
          <div>
            <p className="text-xs text-blue-800">
              {metrics.knowledge_gain > 0.7
                ? "Great progress! Try exploring related topics to expand your knowledge."
                : metrics.engagement_level > 0.7
                ? "You're highly engaged! This is the perfect time to tackle challenging concepts."
                : "Ask follow-up questions to deepen your understanding of the topic."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
