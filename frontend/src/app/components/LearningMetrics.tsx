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
  if (!metrics) return null;

  const MetricItem = ({ label, value }: { label: string; value: number }) => (
    <div className="flex flex-col space-y-1">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 to-indigo-600 transition-all duration-500"
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
      </div>
      <span className="text-xs text-gray-500">{(value * 100).toFixed(0)}%</span>
    </div>
  );

  return (
    <div className="space-y-4 bg-white/50 backdrop-blur-sm rounded-xl p-4 border border-gray-100">
      <h3 className="text-sm font-medium text-gray-800">Learning Progress</h3>
      <div className="space-y-3">
        <MetricItem label="Knowledge Gain" value={metrics.knowledge_gain} />
        <MetricItem label="Engagement" value={metrics.engagement_level} />
        <MetricItem label="Performance" value={metrics.performance_score} />
        <MetricItem
          label="Strategy Effectiveness"
          value={metrics.strategy_effectiveness}
        />
      </div>
    </div>
  );
}
