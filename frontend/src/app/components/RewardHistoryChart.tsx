"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface RewardDataPoint {
  timestamp: string;
  reward: number;
  action?: string;
  displayTime?: string;
}

interface RewardHistoryChartProps {
  data: RewardDataPoint[];
  height?: number;
  showReferenceLine?: boolean;
}

export default function RewardHistoryChart({
  data,
  height = 200,
  showReferenceLine = true,
}: RewardHistoryChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50 rounded-lg p-4 text-gray-500">
        <p className="text-sm">No reward history data available</p>
      </div>
    );
  }

  // Process data to add display time
  const processedData = data.map((point) => ({
    ...point,
    displayTime:
      point.displayTime ||
      new Date(point.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
  }));

  // Get min and max rewards for better axis visualization
  const minReward = Math.min(...processedData.map((d) => d.reward));
  const maxReward = Math.max(...processedData.map((d) => d.reward));

  // Add some padding to the domain
  const yDomain = [
    Math.floor(Math.min(minReward, 0) * 1.1),
    Math.ceil(Math.max(maxReward, 0) * 1.1),
  ];

  return (
    <div className="w-full h-full" style={{ height: height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={processedData}
          margin={{ top: 10, right: 10, left: 0, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="displayTime"
            stroke="#6b7280"
            fontSize={11}
            tick={{ fill: "#6b7280" }}
            tickMargin={8}
          />
          <YAxis
            stroke="#6b7280"
            fontSize={11}
            domain={yDomain}
            tick={{ fill: "#6b7280" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "white",
              borderRadius: "0.375rem",
              border: "1px solid #e5e7eb",
              fontSize: "0.75rem",
              boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
            }}
            formatter={(value: number) => [`${value.toFixed(2)}`, "Reward"]}
            labelFormatter={(label) => `Time: ${label}`}
          />
          {showReferenceLine && (
            <ReferenceLine y={0} stroke="#9CA3AF" strokeDasharray="3 3" />
          )}
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#8B5CF6"
            strokeWidth={2}
            dot={{
              r: 3,
              stroke: "#8B5CF6",
              strokeWidth: 1,
              fill: "white",
            }}
            activeDot={{
              r: 5,
              stroke: "#8B5CF6",
              strokeWidth: 1,
              fill: "#A78BFA",
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
