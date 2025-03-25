"use client";

import { useState } from "react";
import { UserState, TeachingStrategy, LearningMetrics } from "../types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import dynamic from "next/dynamic";

const DetailedMetricsPanel = dynamic(() => import("./DetailedMetricsPanel"), {
  loading: () => <div>Loading detailed metrics...</div>,
});

const TopicKnowledgeGraph = dynamic(() => import("./TopicKnowledgeGraph"), {
  loading: () => <div>Loading knowledge graph...</div>,
});

interface EnhancedMetrics extends LearningMetrics {
  rl_stats: {
    policy_loss: number;
    value_loss: number;
    entropy: number;
    learning_rate: number;
    success_rate: number;
  };
  adaptive_metrics: {
    strategy_adaptation_rate: number;
    response_quality_trend: number[];
    context_relevance: number;
    personalization_score: number;
  };
}

interface AdvancedMetricsProps {
  userState: UserState;
  currentStrategy: TeachingStrategy | null;
  learningHistory: Array<{
    timestamp: string;
    knowledge: number;
    engagement: number;
    performance: number;
  }>;
  currentMetrics: LearningMetrics | null;
  detailedMetrics?: {
    topicMastery: Record<string, number>;
    learningSpeed: number;
    interactionQuality: number;
    contextUtilization: number;
  };
  enhancedMetrics?: EnhancedMetrics;
}

export default function AdvancedMetrics({
  userState,
  currentStrategy,
  learningHistory,
  currentMetrics,
  detailedMetrics,
  enhancedMetrics,
}: AdvancedMetricsProps) {
  const [activeTab, setActiveTab] = useState("progress");

  const tabs = [
    { id: "progress", label: "Learning Progress" },
    { id: "engagement", label: "Engagement Metrics" },
    { id: "strategy", label: "Teaching Strategy" },
    { id: "session", label: "Session Stats" },
    { id: "detailed", label: "Detailed Metrics" },
  ];

  const getMetricColor = (value: number) => {
    if (value >= 0.7) return "text-green-500";
    if (value >= 0.4) return "text-yellow-500";
    return "text-red-500";
  };

  const renderDetailedMetrics = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <DetailedMetricsPanel
        metrics={detailedMetrics}
        learningHistory={learningHistory}
      />
      <TopicKnowledgeGraph
        userId={userState?.user_id}
      />
    </div>
  );

  const renderRLMetrics = () => {
    if (!enhancedMetrics?.rl_stats) return null;

    const { rl_stats } = enhancedMetrics;
    return (
      <div className="p-4 bg-violet-50 dark:bg-violet-900/30 rounded-lg">
        <h4 className="font-medium mb-2">RL Performance</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Policy Loss</span>
              <span className={getMetricColor(1 - rl_stats.policy_loss)}>
                {rl_stats.policy_loss.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Value Loss</span>
              <span className={getMetricColor(1 - rl_stats.value_loss)}>
                {rl_stats.value_loss.toFixed(4)}
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Entropy</span>
              <span>{rl_stats.entropy.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Success Rate</span>
              <span className={getMetricColor(rl_stats.success_rate)}>
                {(rl_stats.success_rate * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <h4 className="font-medium mb-2">Adaptation Metrics</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Strategy Adaptation</span>
              <span
                className={getMetricColor(
                  enhancedMetrics.adaptive_metrics.strategy_adaptation_rate
                )}
              >
                {(
                  enhancedMetrics.adaptive_metrics.strategy_adaptation_rate *
                  100
                ).toFixed(1)}
                %
              </span>
            </div>
            <div className="flex justify-between">
              <span>Personalization</span>
              <span
                className={getMetricColor(
                  enhancedMetrics.adaptive_metrics.personalization_score
                )}
              >
                {(
                  enhancedMetrics.adaptive_metrics.personalization_score * 100
                ).toFixed(1)}
                %
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg space-y-4">
      {/* Tab Navigation */}
      <div className="flex space-x-2 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg transition-colors whitespace-nowrap ${
              activeTab === tab.id
                ? "bg-blue-500 text-white"
                : "bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Progress Metrics */}
      {activeTab === "progress" && (
        <div className="space-y-4">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={learningHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="knowledge"
                  stroke="#3B82F6"
                  name="Knowledge"
                />
                <Line
                  type="monotone"
                  dataKey="performance"
                  stroke="#10B981"
                  name="Performance"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {currentMetrics && (
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
                <h4 className="font-medium mb-2">Current Session</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Knowledge Gain</span>
                    <span
                      className={getMetricColor(currentMetrics.knowledge_gain)}
                    >
                      {(currentMetrics.knowledge_gain * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Performance</span>
                    <span
                      className={getMetricColor(
                        currentMetrics.performance_score
                      )}
                    >
                      {(currentMetrics.performance_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
                <h4 className="font-medium mb-2">Overall Progress</h4>
                <div className="text-2xl font-bold">
                  {(userState.knowledge_level * 100).toFixed(1)}%
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Learning Rate:{" "}
                  {userState.session_metrics.learning_rate.toFixed(2)}/hr
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Engagement Metrics */}
      {activeTab === "engagement" && (
        <div className="space-y-4">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={userState.session_metrics.engagement_trend.map(
                  (value, index) => ({
                    time: index,
                    engagement: value,
                  })
                )}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="engagement"
                  stroke="#8B5CF6"
                  fill="#8B5CF6"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-purple-50 dark:bg-purple-900/30 rounded-lg">
              <h4 className="font-medium mb-2">Session Stats</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Messages</span>
                  <span>{userState.session_metrics.messages_count}</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Response Time</span>
                  <span>
                    {userState.session_metrics.average_response_time.toFixed(1)}
                    s
                  </span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-indigo-50 dark:bg-indigo-900/30 rounded-lg">
              <h4 className="font-medium mb-2">Topics Covered</h4>
              <div className="flex flex-wrap gap-2">
                {userState.session_metrics.topics_covered.map(
                  (topic, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-indigo-100 dark:bg-indigo-800 rounded-full text-sm"
                    >
                      {topic}
                    </span>
                  )
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Strategy Analysis */}
      {activeTab === "strategy" && currentStrategy && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-yellow-50 dark:bg-yellow-900/30 rounded-lg">
              <h4 className="font-medium mb-2">Current Approach</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Style</span>
                  <span className="font-medium">{currentStrategy.style}</span>
                </div>
                <div className="flex justify-between">
                  <span>Complexity</span>
                  <span className="font-medium">
                    {currentStrategy.complexity}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Examples</span>
                  <span className="font-medium">
                    {currentStrategy.examples}
                  </span>
                </div>
              </div>
            </div>

            {currentMetrics && (
              <div className="p-4 bg-orange-50 dark:bg-orange-900/30 rounded-lg">
                <h4 className="font-medium mb-2">Strategy Effectiveness</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Overall</span>
                    <span
                      className={getMetricColor(
                        currentMetrics.strategy_effectiveness
                      )}
                    >
                      {(currentMetrics.strategy_effectiveness * 100).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Engagement</span>
                    <span
                      className={getMetricColor(
                        currentMetrics.engagement_level
                      )}
                    >
                      {(currentMetrics.engagement_level * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="p-4 bg-gray-50 dark:bg-gray-700/30 rounded-lg">
            <h4 className="font-medium mb-2">Strategy Evolution</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              The teaching strategy adapts based on your learning patterns and
              engagement levels. Current focus is on {currentStrategy.style}{" "}
              approach with {currentStrategy.complexity} complexity.
            </p>
          </div>
        </div>
      )}

      {/* Session Stats */}
      {activeTab === "session" && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <h4 className="font-medium mb-2">Session Overview</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span>Duration</span>
                  <span>
                    {Math.floor(
                      (userState.session_metrics.messages_count *
                        userState.session_metrics.average_response_time) /
                        60
                    )}{" "}
                    min
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Topics Covered</span>
                  <span>{userState.session_metrics.topics_covered.length}</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
              <h4 className="font-medium mb-2">Learning Impact</h4>
              <div className="space-y-2">
                {currentMetrics && (
                  <div className="flex justify-between">
                    <span>Quality Score</span>
                    <span
                      className={getMetricColor(
                        currentMetrics.interaction_quality
                      )}
                    >
                      {(currentMetrics.interaction_quality * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span>Topics Mastered</span>
                  <span>{userState.recent_topics.length}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Detailed Metrics */}
      {activeTab === "detailed" && (
        <div className="space-y-4">
          {renderDetailedMetrics()}
          {renderRLMetrics()}
        </div>
      )}

      {/* Real-time metrics updates */}
      {enhancedMetrics && (
        <div className="fixed bottom-4 right-4 space-y-2">
          <div className="bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg">
            Learning Rate:{" "}
            {(enhancedMetrics.rl_stats.learning_rate * 100).toFixed(1)}%
          </div>
          <div className="bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg">
            Adaptation Score:{" "}
            {(
              enhancedMetrics.adaptive_metrics.strategy_adaptation_rate * 100
            ).toFixed(1)}
            %
          </div>
        </div>
      )}
    </div>
  );
}
