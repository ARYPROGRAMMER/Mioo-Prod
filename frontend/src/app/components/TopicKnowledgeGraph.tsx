"use client";

import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

interface TopicKnowledgeGraphProps {
  userId?: string;
  topics?: string[];
  mastery?: Record<string, number>;
}

export default function TopicKnowledgeGraph({
  userId,
  topics,
  mastery,
}: TopicKnowledgeGraphProps) {
  const [data, setData] = useState<Array<{ topic: string; mastery: number }>>(
    []
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // If we have direct mastery data passed in, use that
    if (mastery && Object.keys(mastery).length > 0) {
      const formattedData = Object.entries(mastery).map(([topic, value]) => ({
        topic: topic.charAt(0).toUpperCase() + topic.slice(1),
        mastery: typeof value === "number" ? value * 100 : 0,
      }));
      setData(formattedData);
      setLoading(false);
      return;
    }

    // Otherwise fetch from API if userId is provided
    const fetchData = async () => {
      if (!userId) {
        setLoading(false);
        return;
      }

      try {
        const apiUrl = `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
        }`;
        const response = await fetch(`${apiUrl}/learning-progress/${userId}`);
        const data = await response.json();

        // Transform data into radar chart format
        const formattedData = Object.entries(data.topics_in_progress || {}).map(
          ([topic, mastery]) => ({
            topic: topic.charAt(0).toUpperCase() + topic.slice(1),
            mastery: typeof mastery === "number" ? mastery * 100 : 0,
          })
        );

        setData(formattedData);
      } catch (error) {
        console.error("Error fetching learning progress:", error);
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchData();
      const interval = setInterval(fetchData, 5 * 60 * 1000);
      return () => clearInterval(interval);
    }
  }, [userId, mastery]);

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center bg-gray-50/50 rounded-xl backdrop-blur-sm">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-indigo-600 border-t-transparent" />
      </div>
    );
  }

  // Handle empty data case
  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center bg-white/50 backdrop-blur-sm rounded-xl p-4 text-gray-500 text-center">
        <div>
          <p className="font-medium mb-2">No topic data available</p>
          <p className="text-sm">Start learning to see your topic mastery</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-64 bg-white/50 backdrop-blur-sm rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-800 mb-2">Topic Mastery</h3>
      <ResponsiveContainer width="100%" height="90%">
        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid strokeDasharray="3 3" stroke="#9ca3af" />
          <PolarAngleAxis
            dataKey="topic"
            tick={{ fill: "#4b5563", fontSize: 12 }}
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#9ca3af" />
          <Radar
            name="Mastery"
            dataKey="mastery"
            stroke="#4f46e5"
            fill="#818cf8"
            fillOpacity={0.4}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
