export class TrainingService {
    private baseUrl = 'http://localhost:8000';
    private ws: WebSocket | null = null;

    async startTraining(episodes: number): Promise<string> {
        const response = await fetch(`${this.baseUrl}/train/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ episodes }),
        });
        
        if (!response.ok) {
            throw new Error('Failed to start training');
        }
        
        const data = await response.json();
        return data.training_id;
    }

    async getTrainingStatus(trainingId: string): Promise<any> {
        const response = await fetch(`${this.baseUrl}/train/status/${trainingId}`);
        if (!response.ok) {
            throw new Error('Failed to get training status');
        }
        return response.json();
    }

    connectToTrainingUpdates(trainingId: string, onUpdate: (data: any) => void): void {
        this.ws = new WebSocket(`ws://localhost:8000/train/ws/${trainingId}`);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onUpdate(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    disconnect(): void {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
