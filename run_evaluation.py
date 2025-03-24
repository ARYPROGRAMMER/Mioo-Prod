import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

from personas.student_personas import PersonaManager, StudentPersona
from models.rl_model import PersonaRLAgent
from trainers.persona_trainer import ContentAction
from utils.content_generator import PersonaContentGenerator
from utils.evaluation_metrics import PersonaAdaptationEvaluator

def plot_evaluation_results(results: Dict[str, Any], output_dir: str):
    """Plot evaluation results and save to output directory."""
    # Plot match score distributions
    categories = ['Overall', 'Style', 'Theme', 'Difficulty']
    averages = [
        results['average_match_score'], 
        results['average_style_match'],
        results['average_theme_match'], 
        results['average_difficulty_match']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, averages, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.ylim(0, 1.0)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7)
    plt.title('Average Match Scores by Category')
    plt.ylabel('Match Score (0-1)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'match_scores.png'))
    plt.close()
    
    # Plot match quality distribution
    match_dist = results['match_distribution']
    labels = ['Excellent', 'Good', 'Moderate', 'Poor']
    sizes = [match_dist['excellent'], match_dist['good'], match_dist['moderate'], match_dist['poor']]
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    explode = (0.1, 0, 0, 0)  # explode the 1st slice (Excellent)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Match Quality')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'match_distribution.png'))
    plt.close()

def generate_evaluation_report(personas: List[StudentPersona], content_plans: List[Dict[str, Any]], 
                              evaluations: List[Dict[str, Any]], output_file: str):
    """Generate detailed evaluation report in markdown format."""
    with open(output_file, 'w') as f:
        f.write("# Persona-Based Content Adaptation Evaluation Report\n\n")
        f.write("## Overview\n\n")
        f.write(f"Total personas evaluated: {len(personas)}\n\n")
        
        # Write summary statistics
        batch_eval = PersonaAdaptationEvaluator().evaluate_batch(
            personas, 
            [ContentAction.from_action_id(plan.get("action_id", 0)) for plan in content_plans]
        )
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- Average Overall Match Score: {batch_eval['average_match_score']:.2f}\n")
        f.write(f"- Average Style Match: {batch_eval['average_style_match']:.2f}\n")
        f.write(f"- Average Theme Match: {batch_eval['average_theme_match']:.2f}\n")
        f.write(f"- Average Difficulty Match: {batch_eval['average_difficulty_match']:.2f}\n\n")
        
        f.write("## Match Quality Distribution\n\n")
        f.write(f"- Excellent matches (≥0.8): {batch_eval['match_distribution']['excellent']}\n")
        f.write(f"- Good matches (0.6-0.79): {batch_eval['match_distribution']['good']}\n")
        f.write(f"- Moderate matches (0.4-0.59): {batch_eval['match_distribution']['moderate']}\n")
        f.write(f"- Poor matches (<0.4): {batch_eval['match_distribution']['poor']}\n\n")
        
        f.write("## Detailed Results by Persona\n\n")
        
        # Write individual persona evaluations
        for i, (persona, plan, eval_result) in enumerate(zip(personas, content_plans, evaluations)):
            f.write(f"### {i+1}. {persona.name}\n\n")
            f.write("#### Persona Details\n\n")
            f.write(f"- Education Level: {persona.education_level}\n")
            f.write(f"- Interests: {persona.likes_theme}\n")
            f.write(f"- Learning Style: {persona.learning_style}\n")
            f.write(f"- Average Math Grade: {_calculate_avg_grade(persona.math_mastery):.1f}\n\n")
            
            f.write("#### Selected Content Strategy\n\n")
            f.write(f"- Teaching Style: {plan.get('content_style', 'N/A')}\n")
            f.write(f"- Difficulty Level: {plan.get('difficulty_level', 'N/A')}\n")
            f.write(f"- Content Theme: {plan.get('theme', 'N/A')}\n\n")
            
            f.write("#### Evaluation\n\n")
            f.write(f"- Overall Match Score: {eval_result.get('overall_match_score', 0):.2f}\n")
            f.write(f"- Style Match: {eval_result.get('style_match', 0):.2f}\n")
            f.write(f"- Theme Match: {eval_result.get('theme_match', 0):.2f}\n")
            f.write(f"- Difficulty Match: {eval_result.get('difficulty_match', 0):.2f}\n\n")
            
            f.write("#### Analysis\n\n")
            f.write(f"- {eval_result.get('style_details', '')}\n")
            f.write(f"- {eval_result.get('theme_details', '')}\n")
            f.write(f"- {eval_result.get('difficulty_details', '')}\n\n")
            f.write("---\n\n")

def _calculate_avg_grade(math_mastery: Dict[str, str]) -> float:
    """Calculate average grade from math mastery data."""
    grades = []
    for subject, grade in math_mastery.items():
        if grade.startswith("Grade "):
            try:
                grades.append(int(grade.split(" ")[1]))
            except ValueError:
                pass
    
    return sum(grades) / len(grades) if grades else 0

def main():
    parser = argparse.ArgumentParser(description='Evaluate persona-based content adaptation')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--personas-file', type=str, help='Path to JSON file with personas (optional)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load personas
    persona_manager = PersonaManager()
    if args.personas_file:
        persona_manager.load_from_file(args.personas_file)
    else:
        # Use the predefined personas from main.py
        with open('persona_data.json', 'r') as f:
            persona_data = f.read()
        persona_manager.load_from_json(persona_data)
    
    personas = persona_manager.personas
    print(f"Loaded {len(personas)} personas for evaluation")
    
    # Load the content generator
    try:
        content_generator = PersonaContentGenerator(args.model)
        print(f"Loaded model from {args.model}")
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Please train the model first using main.py")
        return
    
    # Generate content plans for each persona
    print("Generating content plans...")
    content_plans = content_generator.batch_generate_plans(personas)
    
    # Initialize evaluator
    evaluator = PersonaAdaptationEvaluator()
    
    # Evaluate each persona-action pair
    print("Evaluating persona matches...")
    evaluations = []
    for persona, plan in zip(personas, content_plans):
        action = ContentAction.from_action_id(content_generator.agent.choose_action(persona.to_dict()))
        plan['action_id'] = action.action_id
        evaluation = evaluator.evaluate_persona_match(persona, action)
        evaluations.append(evaluation)
    
    # Calculate batch statistics
    batch_evaluation = evaluator.evaluate_batch(
        personas,
        [ContentAction.from_action_id(plan.get('action_id', 0)) for plan in content_plans]
    )
    
    # Generate visualizations
    print("Generating evaluation visualizations...")
    plot_evaluation_results(batch_evaluation, args.output_dir)
    
    # Generate report
    report_file = os.path.join(args.output_dir, 'evaluation_report.md')
    print(f"Generating evaluation report: {report_file}")
    generate_evaluation_report(personas, content_plans, evaluations, report_file)
    
    # Save raw results as JSON
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'summary': batch_evaluation,
            'individual_evaluations': [eval_dict for eval_dict in evaluations]
        }, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"Overall match score: {batch_evaluation['average_match_score']:.2f}")
    
    # Print distribution of match quality
    print("\nMatch quality distribution:")
    for category, count in batch_evaluation['match_distribution'].items():
        print(f"  {category.capitalize()}: {count} personas ({count/len(personas)*100:.1f}%)")

if __name__ == "__main__":
    main()
