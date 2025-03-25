"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
} from "recharts";

interface DetailedMetricsPanelProps {
  metrics:
    | {
        topicMastery: Record<string, number>;
        learningSpeed: number;
        interactionQuality: number;
        contextUtilization: number;
      }
    | undefined;
  learningHistory: Array<{
    timestamp: string;
    knowledge: number;
    engagement: number;
    performance: number;
  }>;
}

export default function DetailedMetricsPanel({
  metrics
}: DetailedMetricsPanelProps) {
  if (!metrics) return null;

  const radarData = [
    {
      metric: "Learning Speed",
      value: metrics.learningSpeed,
    },
    {
      metric: "Quality",
      value: metrics.interactionQuality,
    },
    {
      metric: "Context",
      value: metrics.contextUtilization,
    },
    {
      metric: "Topic Mastery",
      value:
        Object.values(metrics.topicMastery).reduce((a, b) => a + b, 0) /
        Math.max(1, Object.keys(metrics.topicMastery).length),
    },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Learning Analytics</h3>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <PolarRadiusAxis domain={[0, 1]} />
            <Radar
              name="Metrics"
              dataKey="value"
              stroke="#8884d8"
              fill="#8884d8"
              fillOpacity={0.6}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        {Object.entries(metrics.topicMastery).map(([topic, mastery]) => (
          <div key={topic} className="bg-gray-50 dark:bg-gray-700 rounded p-2">
            <div className="text-sm font-medium">{topic}</div>
            <div className="mt-1 h-2 bg-gray-200 rounded">
              <div
                className="h-2 bg-blue-500 rounded"
                style={{ width: `${mastery * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
