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

export interface EnhancedMetrics extends LearningMetrics {
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

export interface ChatResponse {
  response: string;
  teaching_strategy: TeachingStrategy;
  metrics: LearningMetrics;
  human_feedback?: "like" | "dislike"; // new field for RLHF feedback
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
  learning_style?: LearningStyle;
  feedback_history?: Array<"like" | "dislike">;
}

export interface MessageAnalysis {
  topic: string;
  complexity: number;
  context: Record<string, any>;
  learning_style: Record<string, number>;
  sentiment: number;
}

export interface EnhancedChatRequest extends ChatRequest {
  analysis?: MessageAnalysis;
}

export interface FeedbackRequest {
  user_id: string;
  message_id: string;
  feedback: "like" | "dislike";
}

export interface FeedbackResponse {
  status: string;
  feedback_processed: boolean;
}

export interface LearningStyle {
  visual: number;
  interactive: number;
  theoretical: number;
  practical: number;
}
