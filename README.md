# Personalized AI Tutor - RL Model

This reinforcement learning system adapts educational content based on high school student personas, specializing in mathematics education.

## Overview

This project implements a reinforcement learning system that learns to adapt teaching content to match student preferences and learning styles. The system specializes in math education for high school students preparing for SAT/ACT exams.

## Features

- Persona-based content adaptation using reinforcement learning
- Multiple teaching styles, difficulty levels, and thematic approaches
- Content generation tailored to individual student preferences
- Detailed evaluation metrics for measuring adaptation quality

## Student Personas

The system is trained on 10 high school student personas:

1. **Ethan**: Sports enthusiast who prefers real-world examples
2. **Olivia**: Video game fan who likes step-by-step problem solving
3. **Mason**: Technology-oriented student who prefers concise instructions
4. **Ava**: Foodie who benefits from visual learning aids
5. **Logan**: Entertainment and media fan who likes detailed examples
6. **Sophia**: Fashion-oriented student who prefers interactive learning
7. **Jackson**: Music enthusiast who understands through analogies
8. **Mia**: Theme park fan who enjoys story-based problems
9. **Lucas**: Automotive enthusiast who likes bullet-point summaries
10. **Isabella**: Photography fan who learns through visual examples

## Getting Started

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### Training the Model

Train the reinforcement learning model on the student personas:

```bash
python main.py --episodes 2000 --output-dir models --evaluate
```

### Generating Content

Generate adaptive content for student personas:

```bash
python content_generation.py --model models/best_model.pth --sample
```

For a specific student:

```bash
python content_generation.py --model models/best_model.pth --persona Ethan --sample
```

### Evaluating Adaptation Quality

Run evaluation metrics on the trained model:

```bash
python run_evaluation.py --model models/best_model.pth --output-dir evaluation_results
```

## System Architecture

- `personas/`: Defines student personas and their attributes
- `models/`: Implements the RL model and neural networks
- `trainers/`: Contains training logic and environment simulation
- `utils/`: Helper utilities for content generation and evaluation
- `templates/`: Default configuration templates

## How It Works

1. The system extracts features from student personas
2. The RL agent selects adaptation actions (style, difficulty, theme)
3. A reward function evaluates how well the action matches the student's preferences
4. The system learns over time to make better adaptation decisions
5. Content is generated based on the selected adaptation approach

## License

This project is provided for educational purposes only.
