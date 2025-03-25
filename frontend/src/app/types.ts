export type QueryType =
  | "educational"
  | "general"
  | "factual"
  | "conversational";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

export interface ChatRequest {
  message: string;
  user_id: string;
}

export interface LearningMetrics {
  knowledge_gain: number;
  engagement_level: number;
  performance_score: number;
  strategy_effectiveness: number;
  interaction_quality: number;
}

export interface ChatResponse {
  response: string;
  teaching_strategy: TeachingStrategy;
  metrics: LearningMetrics;
}

export interface TeachingStrategy {
  style: string;
  complexity: string;
  examples: string;
}

export interface SessionMetrics {
  messages_count: number;
  average_response_time: number;
  topics_covered: string[];
  learning_rate: number;
  engagement_trend: number[];
}

export interface LearningHistoryEntry {
  timestamp: string;
  knowledge: number;
  engagement: number;
  performance: number;
  strategy: TeachingStrategy;
}

export interface UserState {
  user_id: string;
  knowledge_level: number;
  engagement: number;
  interests: string[];
  recent_topics: string[];
  performance: number;
  chat_history: Message[];
  last_updated: string;
  learning_history: LearningHistoryEntry[];
  session_metrics: SessionMetrics;
}

export interface MessageAnalysis {
  topic: string;
  complexity: number;
  context: Record<string, string>;
  learning_style: Record<string, number>;
  sentiment: number;
}

export interface EnhancedChatRequest extends ChatRequest {
  analysis?: MessageAnalysis;
}
