import { ChatRequest, ChatResponse, UserState } from '../app/types';

export class APIService {
    private baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async getUserState(userId: string): Promise<UserState> {
        const response = await fetch(`${this.baseUrl}/user/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }

    async getLearningProgress(userId: string): Promise<any> {
        const response = await fetch(`${this.baseUrl}/learning-progress/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }
}

export const apiService = new APIService();
