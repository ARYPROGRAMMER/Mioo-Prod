# Mioo AI Tutor

An adaptive learning platform powered by reinforcement learning and large language models.

## Overview

Mioo AI Tutor is an educational platform that adapts to each learner's unique needs. It uses reinforcement learning to optimize teaching strategies and leverages OpenAI's GPT-4o to provide personalized educational content.

## Architecture

The system consists of three main components:

1. **Reinforcement Learning System**: A PPO (Proximal Policy Optimization) agent that selects the optimal teaching strategy based on the user's state.
2. **Large Language Model Integration**: Uses OpenAI's GPT-4o to generate educational content tailored to the selected strategy.
3. **User State Management**: Tracks user knowledge, engagement, and learning history to inform the RL agent.

## Key Features

- **Adaptive Teaching Strategies**: Automatically selects from different teaching styles (detailed, concise, interactive, analogy-based, step-by-step).
- **Personalized Complexity**: Adjusts the complexity level based on the user's knowledge.
- **Dynamic Learning Assessment**: Continuously evaluates knowledge gain and engagement.
- **Topic Mastery Tracking**: Visualizes progress across different topics.
- **Feedback Integration**: Uses explicit feedback to improve the RL model.

## Technical Stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, Recharts
- **Backend**: FastAPI, Python 3.10+
- **ML/RL**: PyTorch, PPO algorithm
- **LLM**: OpenAI API (GPT-4o)
- **Database**: MongoDB with Motor for async operations

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 16+
- MongoDB
- OpenAI API key

### Backend Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```
   export OPENAI_API_KEY="your-api-key"
   export MONGODB_URI="mongodb://localhost:27017"
   export MONGODB_DB="mioo_tutor"
   ```
4. Run the backend:
   ```
   python main.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```
2. Install dependencies:
   ```
   npm install
   ```
3. Set up environment variables (create `.env.local`):
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
4. Run the frontend:
   ```
   npm run dev
   ```

## API Endpoints

- `POST /chat`: Send a message to the AI tutor
- `GET /user/{user_id}`: Get user state
- `PUT /user/{user_id}`: Update user preferences
- `POST /feedback`: Submit feedback for a message
- `GET /learning-progress/{user_id}`: Get detailed learning progress

## System Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Frontend   │◄────►│  FastAPI API  │◄────►│   MongoDB    │
└──────────────┘      └──────────────┘      └──────────────┘
                            │  ▲
                            ▼  │
                     ┌──────────────┐
                     │  RL System   │
                     └──────────────┘
                            │  ▲
                            ▼  │
                     ┌──────────────┐
                     │  OpenAI API  │
                     └──────────────┘
```

## License

MIT
