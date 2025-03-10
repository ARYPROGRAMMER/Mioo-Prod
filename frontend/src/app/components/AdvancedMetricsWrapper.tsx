"use client";

import { useState } from "react";
import AdvancedMetrics from "./AdvancedMetrics";
import {
  UserState,
  TeachingStrategy,
  LearningMetrics,
  EnhancedMetrics,
} from "../types";
import RewardHistoryChart from "./RewardHistoryChart";

interface AdvancedMetricsWrapperProps {
  userState: UserState;
  currentStrategy: TeachingStrategy | null;
  learningHistory: any[];
  currentMetrics: LearningMetrics | null;
  detailedMetrics?: {
    topicMastery: Record<string, number>;
    learningSpeed: number;
    interactionQuality: number;
    contextUtilization: number;
  };
}

export default function AdvancedMetricsWrapper({
  userState,
  currentStrategy,
  learningHistory,
  currentMetrics,
  detailedMetrics,
}: AdvancedMetricsWrapperProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Mock data for reward history chart
  const rewardData = learningHistory.map((item, index) => ({
    timestamp: item.timestamp,
    reward:
      ((item.knowledge || 0.5) * 0.6 + (item.engagement || 0.5) * 0.4 - 0.5) *
      2,
    action: "learn",
  }));

  // Create enhanced metrics only if currentMetrics exists
  const enhancedMetrics: EnhancedMetrics | undefined = currentMetrics
    ? {
        // Include all existing metrics properties
        knowledge_gain: currentMetrics.knowledge_gain,
        engagement_level: currentMetrics.engagement_level,
        performance_score: currentMetrics.performance_score,
        strategy_effectiveness: currentMetrics.strategy_effectiveness,
        interaction_quality: currentMetrics.interaction_quality,

        // Add RL stats
        rl_stats: {
          policy_loss: 0.23,
          value_loss: 0.18,
          entropy: 0.35,
          learning_rate: 0.6,
          success_rate: 0.75,
        },

        // Add adaptive metrics
        adaptive_metrics: {
          strategy_adaptation_rate: 0.65,
          response_quality_trend: [0.6, 0.65, 0.7, 0.75],
          context_relevance: 0.8,
          personalization_score: 0.7,
        },
      }
    : undefined;

  if (!showAdvanced) {
    return (
      <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4 mt-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-sm font-medium text-gray-700">
            Learning Overview
          </h3>
          <button
            onClick={() => setShowAdvanced(true)}
            className="text-xs bg-indigo-50 text-indigo-600 py-1 px-2 rounded-md hover:bg-indigo-100"
          >
            Show Advanced Metrics
          </button>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-50 rounded-md p-4">
            <h4 className="text-xs font-medium text-gray-500 mb-2">
              Knowledge Progress
            </h4>
            <div className="flex justify-between items-center">
              <div>
                <span className="text-lg font-medium text-gray-800">
                  {(userState.knowledge_level * 100).toFixed(0)}%
                </span>
                {currentMetrics && (
                  <span className="ml-2 text-xs bg-green-50 text-green-700 px-1.5 py-0.5 rounded">
                    +{(currentMetrics.knowledge_gain * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
          </div>

          {rewardData.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-500 mb-2">
                Reward Trend
              </h4>
              <div className="h-32 bg-white rounded-md border border-gray-100">
                <RewardHistoryChart data={rewardData.slice(-5)} height={120} />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4 mt-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-medium text-gray-700">
          Advanced Learning Metrics
        </h3>
        <button
          onClick={() => setShowAdvanced(false)}
          className="text-xs bg-gray-100 text-gray-600 py-1 px-2 rounded-md hover:bg-gray-200"
        >
          Show Simple View
        </button>
      </div>

      <AdvancedMetrics
        userState={userState}
        currentStrategy={currentStrategy}
        learningHistory={learningHistory}
        currentMetrics={currentMetrics}
        detailedMetrics={detailedMetrics}
        enhancedMetrics={enhancedMetrics}
      />
    </div>
  );
}
