import argparse
import os
import json
import sys
from typing import Dict, List, Any
from personas.student_personas import PersonaManager, StudentPersona
from utils.content_generator import PersonaContentGenerator

def load_personas(file_path: str):
    """Load personas from a JSON file."""
    manager = PersonaManager()
    manager.load_from_file(file_path)
    return manager.personas

def print_persona_info(persona: StudentPersona):
    """Print formatted information about a persona."""
    print(f"\n{'='*80}")
    print(f"STUDENT PERSONA: {persona.name}")
    print(f"{'='*80}")
    print(f"Education Level: {persona.education_level}")
    print(f"SAT/ACT Timeline: {persona.sat_act_timeline}")
    print("\nMath Mastery Levels:")
    for subject, level in persona.math_mastery.items():
        print(f"  - {subject}: {level}")
    print(f"\nInterests/Theme: {persona.likes_theme}")
    print(f"Learning Style: {persona.learning_style}")
    print(f"{'='*80}")

def print_content_plan(plan: Dict[str, Any]):
    """Print formatted content plan."""
    print("\nCONTENT ADAPTATION PLAN")
    print(f"{'='*80}")
    print(f"Teaching Style: {plan['content_style']}")
    print(f"Difficulty Level: {plan['difficulty_level']}")
    print(f"Content Theme: {plan['theme']}")
    print("\nTheme-Based Examples:")
    for example in plan.get('theme_examples', []):
        print(f"  - {example}")
    
    learning_approach = plan.get('learning_approach', {})
    print("\nLearning Approach:")
    for key, value in learning_approach.items():
        print(f"  - {key.replace('_', ' ').title()}: {value}")
    
    math_level = plan.get('math_level', {})
    print("\nMath Level:")
    print(f"  - Level: {math_level.get('level', 'Unknown')}")
    print(f"  - Average Grade: {math_level.get('avg_grade', 'Unknown')}")
    print("  - Prerequisites:")
    for prereq in math_level.get('prerequisites', []):
        print(f"    * {prereq}")
    print(f"{'='*80}")

def generate_sample_content(plan: Dict[str, Any]) -> str:
    """Generate a sample content snippet based on the plan."""
    style = plan['content_style']
    difficulty = plan['difficulty_level']
    theme = plan['theme']
    example = plan.get('theme_examples', [""])[0]
    
    # Generate a template snippet based on the plan
    content = f"# Sample Content for {plan['student']}\n\n"
    
    if style == "Detailed step-by-step":
        content += "## Quadratic Functions: Step-by-Step Guide\n\n"
        content += "Let's solve this step-by-step:\n\n"
        content += "1. First, we identify the standard form of a quadratic equation: ax² + bx + c = 0\n"
        content += "2. Then, we identify the coefficients a, b, and c\n"
        content += "3. Next, we can use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a\n"
        content += "4. Let's substitute our values and calculate...\n\n"
        
    elif style == "Visual aid focused":
        content += "## Visualizing Quadratic Functions\n\n"
        content += "[Insert graph of parabola here]\n\n"
        content += "Notice how the graph creates a U-shape called a parabola.\n"
        content += "The key parts of this visual are:\n"
        content += "- Vertex: The lowest/highest point\n"
        content += "- Axis of symmetry: The vertical line through the vertex\n"
        content += "- x-intercepts: Where the parabola crosses the x-axis (the solutions!)\n"
        
    elif style == "Real-world application":
        content += f"## Quadratic Functions in {theme}\n\n"
        content += f"Let's explore how quadratic functions appear in {theme}:\n\n"
        content += f"{example}\n\n"
        content += "This real-world scenario can be modeled with a quadratic equation where:\n"
        content += "- The independent variable represents...\n"
        content += "- The dependent variable represents...\n"
    
    elif style == "Simplified language":
        content += "## Quadratic Functions Made Simple\n\n"
        content += "A quadratic function is just a fancy way of saying 'a function with an x²'.\n\n"
        content += "These functions always make U-shaped graphs (either ∪ or ∩).\n\n"
        content += "The basic idea is that we're looking at how one value changes when another value is squared.\n\n"
        
    elif style == "Bullet point summary":
        content += "## Quadratic Functions: Key Points\n\n"
        content += "* Standard form: f(x) = ax² + bx + c\n"
        content += "* Graph is called a parabola (U-shaped)\n"
        content += "* The coefficient 'a' determines if it opens up (a > 0) or down (a < 0)\n"
        content += "* Vertex formula: x = -b/(2a)\n"
        content += "* Solutions can be found using the quadratic formula\n"
        
    elif style == "Interactive questioning":
        content += "## Let's Explore Quadratic Functions\n\n"
        content += "What happens to the graph of f(x) = ax² + bx + c if we change 'a' to a negative number?\n\n"
        content += "[Think about this for a moment]\n\n"
        content += "That's right! The parabola flips and opens downward.\n\n"
        content += "Now, what if we change the value of 'c'? How does that affect the graph?\n"
        
    elif style == "Analogy-based":
        content += "## Understanding Quadratics Through Analogies\n\n"
        content += "Think of a quadratic function like throwing a ball into the air.\n\n"
        content += "• The path the ball follows is a parabola\n"
        content += "• Gravity is like the coefficient 'a' (always making the ball come back down)\n"
        content += "• Your initial force is like the linear term 'b'\n"
        content += "• Your starting height is like the constant 'c'\n"
        
    else:  # Story context or fallback
        content += "## The Story of Quadratic Functions\n\n"
        content += "Imagine you're designing a water fountain display for a theme park.\n"
        content += "The water needs to shoot up and create a perfect arc before landing in a target area.\n\n"
        content += "This is where quadratic functions come in - they describe the exact path the water will follow!\n\n"

    # Add a difficulty-based example
    content += f"\n## {difficulty} Practice Example\n\n"
    
    if difficulty == "Basic":
        content += "Solve for x: x² - 5x + 6 = 0"
    elif difficulty == "Standard":
        content += "Find the vertex of f(x) = 2x² - 8x + 7"
    elif difficulty == "Advanced":
        content += "For what values of k does the equation x² + kx + 4 = 0 have exactly one solution?"
    else:  # Challenge
        content += "Prove that if p is a quadratic polynomial and p(0) = p(1) = p(2) = 0, then p(x) = ax(x-1)(x-2) for some constant a."
        
    return content

def main():
    parser = argparse.ArgumentParser(description='Generate personalized content for student personas')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--persona', type=str, help='Name of specific persona to generate content for')
    parser.add_argument('--output-dir', type=str, default='generated_content', help='Directory to save generated content')
    parser.add_argument('--personas-file', type=str, default='persona_data.json', help='Path to JSON file with personas')
    parser.add_argument('--sample', action='store_true', help='Generate sample content based on the plan')
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train the model first using main.py")
        sys.exit(1)
    
    # Check if personas file exists
    if not os.path.exists(args.personas_file):
        print(f"Error: Personas file not found at {args.personas_file}")
        print("Please create the personas file first")
        sys.exit(1)
    
    # Create output directory if needed
    if args.sample:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load personas
        personas = load_personas(args.personas_file)
        
        if len(personas) == 0:
            print("Error: No personas found in the provided file")
            sys.exit(1)
            
        # Load the content generator
        content_generator = PersonaContentGenerator(args.model)
        
        # Filter to specific persona if requested
        if args.persona:
            personas = [p for p in personas if p.name.lower() == args.persona.lower()]
            if not personas:
                print(f"Error: No persona found with name '{args.persona}'")
                available_names = [p.name for p in load_personas(args.personas_file)]
                print(f"Available personas: {', '.join(available_names)}")
                sys.exit(1)
        
        # Generate content plans
        for persona in personas:
            print_persona_info(persona)
            plan = content_generator.generate_content_plan(persona)
            print_content_plan(plan)
            
            if args.sample:
                content = generate_sample_content(plan)
                
                # Save to file
                filename = f"{persona.name.lower().replace(' ', '_')}_content.md"
                filepath = os.path.join(args.output_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(content)
                    
                print(f"\nSample content saved to: {filepath}")
                
            print("\n" + "-"*40 + "\n")
            
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
