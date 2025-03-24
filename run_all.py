import os
import subprocess
import argparse
import time
import shutil
from datetime import datetime

def run_command(cmd, description=None):
    """Run a command with a description."""
    if description:
        print(f"\n{'='*80}\n{description}\n{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        return False
    
    print(f"\nCommand completed in {elapsed:.1f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the complete RL model pipeline')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, default='output', help='Main output directory')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--clean', action='store_true', help='Clean previous output')
    args = parser.parse_args()
    
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # Create output directories
    model_dir = os.path.join(run_dir, "models")
    content_dir = os.path.join(run_dir, "content")
    eval_dir = os.path.join(run_dir, "evaluation")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Copy persona file to output
    shutil.copy("persona_data.json", os.path.join(run_dir, "persona_data.json"))
    
    # Clean if requested
    if args.clean and os.path.exists("models"):
        print("Cleaning previous output...")
        shutil.rmtree("models", ignore_errors=True)
    
    # Train model
    if not args.skip_training:
        success = run_command(
            f"python main.py --episodes {args.episodes} --output-dir {model_dir} --evaluate",
            "Training RL model on student personas"
        )
        if not success:
            print("Training failed, exiting.")
            return
        
        # Copy training results to models directory for other scripts to find
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)
        for file in os.listdir(model_dir):
            if file.endswith(".pth"):
                shutil.copy(os.path.join(model_dir, file), os.path.join("models", file))
    
    # Run evaluation
    run_command(
        f"python run_evaluation.py --model {model_dir}/best_model.pth --output-dir {eval_dir}",
        "Evaluating model performance"
    )
    
    # Generate content for each persona
    run_command(
        f"python content_generation.py --model {model_dir}/best_model.pth --output-dir {content_dir} --sample",
        "Generating personalized content"
    )
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed successfully!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
