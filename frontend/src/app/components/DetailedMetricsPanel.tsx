"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

interface DetailedMetricsPanelProps {
  metrics: {
    topicMastery: Record<string, number>;
    learningSpeed: number;
    interactionQuality: number;
    contextUtilization: number;
  } | null;
  learningHistory: Array<{
    timestamp: string;
    knowledge: number;
    engagement: number;
    performance: number;
  }>;
}

export default function DetailedMetricsPanel({
  metrics,
  learningHistory,
}: DetailedMetricsPanelProps) {
  if (!metrics) {
    return (
      <div className="flex items-center justify-center h-full bg-white/80 backdrop-blur-sm rounded-lg p-4">
        <div className="flex flex-col items-center text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-600 border-t-transparent mb-3"></div>
          <p className="text-gray-500">Loading metrics data...</p>
        </div>
      </div>
    );
  }

  const topicMastery = metrics.topicMastery || {};
  const avgMastery =
    Object.keys(topicMastery).length > 0
      ? Object.values(topicMastery).reduce((a: number, b: number) => a + b, 0) /
        Object.keys(topicMastery).length
      : 0;

  const radarData = [
    { metric: "Learning Speed", value: metrics.learningSpeed || 0 },
    { metric: "Quality", value: metrics.interactionQuality || 0 },
    { metric: "Context", value: metrics.contextUtilization || 0 },
    { metric: "Topic Mastery", value: avgMastery },
  ];

  // Format timestamp for better display
  const formattedHistory = learningHistory.map((entry) => ({
    ...entry,
    displayTime: new Date(entry.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <div className="space-y-6 h-full overflow-y-auto">
      <div className="border-b border-gray-200 pb-3">
        <h3 className="text-base font-semibold text-gray-800 flex items-center gap-2">
          <span className="w-2 h-2 bg-violet-500 rounded-full"></span>
          Learning Analytics
        </h3>
      </div>

      {/* Charts Section */}
      <div className="grid gap-6">
        {/* Radar Chart */}
        <div className="bg-gray-50 rounded-lg p-4 shadow-sm border border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Performance Metrics
          </h4>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="#e5e7eb" />
                <PolarAngleAxis
                  dataKey="metric"
                  stroke="#6b7280"
                  fontSize={12}
                />
                <PolarRadiusAxis
                  domain={[0, 1]}
                  stroke="#6b7280"
                  tickCount={5}
                />
                <Radar
                  name="Metrics"
                  dataKey="value"
                  stroke="#6366f1"
                  fill="#818cf8"
                  fillOpacity={0.4}
                />
                <Tooltip
                  formatter={(value: number) => [
                    (value * 100).toFixed(0) + "%",
                    "Score",
                  ]}
                  labelFormatter={(label) => `${label}`}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Progress History */}
        {formattedHistory?.length > 0 ? (
          <div className="bg-gray-50 rounded-lg p-4 shadow-sm border border-gray-100">
            <h4 className="text-sm font-medium text-gray-700 mb-3">
              Progress Timeline
            </h4>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={formattedHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="displayTime"
                    stroke="#6b7280"
                    fontSize={11}
                    tickMargin={5}
                  />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={11}
                    domain={[0, 1]}
                    tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    formatter={(value: number) => [
                      (value * 100).toFixed(1) + "%",
                    ]}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="knowledge"
                    stroke="#6366f1"
                    name="Knowledge"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="engagement"
                    stroke="#10b981"
                    name="Engagement"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="performance"
                    stroke="#f59e0b"
                    name="Performance"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="bg-gray-50 rounded-lg p-4 shadow-sm border border-gray-100">
            <h4 className="text-sm font-medium text-gray-700 mb-2">
              Progress Timeline
            </h4>
            <div className="h-40 flex items-center justify-center text-gray-500 text-center">
              <div>
                <p className="mb-1">No learning history available yet</p>
                <p className="text-xs">
                  Continue your interactions to track progress
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Topic Mastery */}
        <div className="bg-gray-50 rounded-lg p-4 shadow-sm border border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Topic Mastery
          </h4>
          {Object.keys(topicMastery).length > 0 ? (
            <div className="grid sm:grid-cols-2 gap-3">
              {Object.entries(topicMastery).map(([topic, mastery]) => (
                <div
                  key={topic}
                  className="bg-white rounded-md p-2 border border-gray-50"
                >
                  <div className="text-xs font-medium text-gray-700">
                    {topic}
                  </div>
                  <div className="mt-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        (mastery || 0) > 0.7
                          ? "bg-green-500"
                          : (mastery || 0) > 0.4
                          ? "bg-yellow-500"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${(mastery || 0) * 100}%` }}
                    />
                  </div>
                  <div className="mt-1 flex justify-between items-center">
                    <div className="text-xs text-gray-500">
                      {((mastery || 0) * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">
                      {(mastery || 0) > 0.8
                        ? "Mastered"
                        : (mastery || 0) > 0.6
                        ? "Proficient"
                        : (mastery || 0) > 0.4
                        ? "Intermediate"
                        : "Beginner"}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6 text-gray-500 bg-white rounded-lg border border-gray-50">
              <p>No topic mastery data available yet</p>
              <p className="text-xs mt-1">
                Start learning to see your progress
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
