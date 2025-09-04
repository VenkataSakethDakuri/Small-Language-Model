"""
Script for end to end training and evaluation of a model with math dataset using gemini LLM as a judge. 
Uses standard and vLLM inference. Uses a range of values and options for hyperparameters for standard LoRA training.
"""

import itertools
import json
import logging
import sys
from typing import Dict, List, Tuple, Any
import os

#Adding GrandParent path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.training.lora_trainer import LoRATrainer
from Modules.inference.standard_inference import StandardInferenceEngine
from Modules.inference.vllm_inference import VLLMInferenceEngine
from Modules.evaluation.geminiJudge import LLMGeminiJudgeEvaluator
from Modules.utils.utlities import MemoryUtils, DataUtils, FileUtils


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_evaluation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_lora_params() -> Dict[str, List]:
    """Define LoRA parameter ranges for hyperparameter search."""
    return {
        "r": [4, 8, 16, 32, 64],
        "lora_alpha": [8, 16, 32, 64, 128],
        "lora_dropout": [0.0, 0.05, 0.1, 0.2],
        "target_modules": [
            ["q_proj", "v_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]
        ],
        "bias": "none"
    }


def get_optimizer_configs() -> Dict[str, Dict[str, List]]:
    """Define all optimizer configurations with their parameter ranges."""
    
    # Direct replacements - same parameter ranges
    direct_replacements = {
        "adamw_8bit": {
            "optim": "adamw_8bit",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 10],
        },
        "adamw_torch": {
            "optim": "adamw_torch",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 10],
        },
        "adamw_bnb_8bit": {
            "optim": "adamw_bnb_8bit",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 10],
        },
        "rmsprop": {
            "optim": "rmsprop",
            "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        }
    }
    
    # Optimizers requiring parameter adjustments
    adjusted_optimizers = {
        "sgd": {
            "optim": "sgd",
            "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["cosine", "linear", "polynomial", "constant_with_warmup"],
            "warmup_steps": [5, 10, 25, 50, 100, 200],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [2, 4, 8, 16, 32],
            "num_train_epochs": [2, 3, 5, 8, 10],
        },
        "lion": {
            "optim": "lion",
            "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4],
            "weight_decay": [0.01, 0.05, 0.1, 0.2, 0.3],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        },
        "adafactor": {
            "optim": "adafactor",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
            "weight_decay": [0.0],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup"],
            "warmup_steps": [0, 5, 10, 25, 50],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        },
        "adagrad": {
            "optim": "adagrad",
            "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        }
    }
    
    # Advanced/Experimental optimizers
    advanced_optimizers = {
        "adamw_apex_fused": {
            "optim": "adamw_apex_fused",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        },
        "adamw_hf": {
            "optim": "adamw_hf",
            "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
            "warmup_steps": [0, 5, 10, 25, 50, 100],
            "per_device_train_batch_size": [1, 2, 4, 8, 16],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        },
        "lamb": {
            "optim": "lamb",
            "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
            "lr_scheduler_type": ["linear", "cosine", "polynomial", "constant_with_warmup"],
            "warmup_steps": [5, 10, 25, 50, 100, 200],
            "per_device_train_batch_size": [2, 4, 8, 16, 32],
            "gradient_accumulation_steps": [1, 2, 4, 8, 16],
            "num_train_epochs": [1, 2, 3, 5, 8],
        }
    }
    
    return {**direct_replacements, **adjusted_optimizers, **advanced_optimizers}


def generate_optimizer_combinations(optimizer_config: Dict[str, List]) -> Tuple[List[str], List[Tuple]]:
    """Generate all combinations for a single optimizer configuration."""
    params = []
    values = []
    
    for param, value_list in optimizer_config.items():
        if param != "optim":  # Skip the optimizer name
            params.append(param)
            values.append(value_list)
    
    combinations = list(itertools.product(*values))
    return params, combinations


def calculate_total_combinations(lora_params: Dict, all_optimizers: Dict) -> Tuple[int, Dict[str, int]]:
    """Calculate total number of combinations and per-optimizer counts."""
    lora_combinations = list(itertools.product(
        lora_params["r"],
        lora_params["lora_alpha"], 
        lora_params["lora_dropout"],
        lora_params["target_modules"]
    ))
    
    total_combinations = 0
    optimizer_counts = {}
    
    for opt_name, opt_config in all_optimizers.items():
        params, combinations = generate_optimizer_combinations(opt_config)
        optimizer_counts[opt_name] = len(combinations)
        total_combinations += len(combinations)
    
    total_with_lora = total_combinations * len(lora_combinations)
    
    return total_with_lora, optimizer_counts


def save_results(result: Dict[str, Any], results_file: str = "hyperparameter_results.json"):
    """Save result to JSON file, appending to existing results."""
    try:
        with open(results_file, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = []
    
    all_results.append(result)
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)


def run_single_experiment(
    lora_config: Dict[str, Any], 
    training_params: Dict[str, Any], 
    config: Dict[str, str], 
    train_data: Any, 
    test_data_path: str, 
    combination_counter: int
) -> Dict[str, Any]:
    """Run a single training and evaluation experiment."""
    
    try:
        # Create trainer instance
        trainer = LoRATrainer(config)
        
        # Setup model with current LoRA parameters
        trainer.setup_model(
            lora_r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"]
        )
        
        # Create unique output directory
        output_dir = f"TrainingMath_LoRA_{combination_counter}"
        training_params["output_dir"] = output_dir
        
        # Train the model
        logging.info(f"Starting training for combination {combination_counter}")
        training_result = trainer.train(train_data, "math", **training_params)
        
        # Run standard inference
        logging.info(f"Running standard inference for combination {combination_counter}")
        inference_engine = StandardInferenceEngine(output_dir, config)
        inference_engine.load_model(base_model=config["model_name"], lora_path=output_dir)
        
        inference_output_path = f"inference_results_{combination_counter}.json"
        inference_results = inference_engine.batch_inference_on_math_dataset(test_data_path, inference_output_path)
        
        # Evaluate standard inference results
        evaluator = LLMGeminiJudgeEvaluator(base_model_name=config["model_name"], config=config)
        evaluation_results = evaluator.evaluate_outputs(inference_output_path)
        average_scores = evaluator.calculate_average_scores(evaluation_results["comparisons"])
        
        MemoryUtils.clear_cache()
        
        # Run vLLM inference
        logging.info(f"Running vLLM inference for combination {combination_counter}")
        vllm_engine = VLLMInferenceEngine(output_dir, config)
        vllm_engine.load_model(base_model=config["model_name"], adapter_path=output_dir)
        
        vllm_output_path = f"vllm_inference_results_{combination_counter}.json"
        vllm_results = vllm_engine.batch_inference_on_dataset(test_data_path, vllm_output_path)
        
        # Evaluate vLLM inference results
        vllm_evaluator = LLMGeminiJudgeEvaluator(base_model_name=config["model_name"], config=config)
        vllm_evaluation_results = vllm_evaluator.evaluate_outputs(vllm_output_path)
        vllm_average_scores = vllm_evaluator.calculate_average_scores(vllm_evaluation_results["comparisons"])
        
        # Cleanup
        vllm_engine.cleanup()
        MemoryUtils.clear_cache()
        
        # Return successful result
        return {
            "combination_id": combination_counter,
            "lora_config": lora_config,
            "optimizer_config": training_params,
            "training_time": training_result["training_time"],
            "peak_memory": training_result["peak_memory"],
            "standard_inference": {
                "finetuned_avg_score": average_scores["finetuned_avg"],
                "base_avg_score": average_scores["base_avg"],
                "total_comparisons": average_scores["total_comparisons"]
            },
            "vllm_inference": {
                "finetuned_avg_score": vllm_average_scores["finetuned_avg"],
                "base_avg_score": vllm_average_scores["base_avg"],
                "total_comparisons": vllm_average_scores["total_comparisons"]
            },
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error in combination {combination_counter}: {str(e)}")
        MemoryUtils.clear_cache()
        return {
            "combination_id": combination_counter,
            "lora_config": lora_config,
            "optimizer_config": training_params,
            "status": "failed",
            "error": str(e)
        }


def main():
    """Main function to run the complete hyperparameter search experiment."""
    
    # Setup logging
    setup_logging()
    logging.info("Starting hyperparameter search experiment")
    
    # Clear memory cache at start
    MemoryUtils.clear_cache()
    
    # Get configurations
    lora_params = get_lora_params()
    all_optimizers = get_optimizer_configs()
    
    # Configuration
    config = {
        "model_name": "openai/gpt-oss-20b"
    }
    
    # Data paths
    train_data_path = r"C:\Users\DELL\OneDrive\Desktop\Project\SLM\Experiments\2\pretraining_data.csv"
    test_data_path = r"C:\Users\DELL\OneDrive\Desktop\Project\SLM\Experiments\2\benchmark_data.csv"
    
    try:
        # Load training data once
        logging.info("Loading training data")
        train_data = DataUtils.load_csv(train_data_path)
        
        # Calculate total combinations
        total_combinations, optimizer_counts = calculate_total_combinations(lora_params, all_optimizers)
        logging.info(f"Total combinations to run: {total_combinations}")
        logging.info(f"Optimizer counts: {optimizer_counts}")
        
        # Generate all LoRA combinations
        lora_combinations = list(itertools.product(
            lora_params["r"],
            lora_params["lora_alpha"], 
            lora_params["lora_dropout"],
            lora_params["target_modules"]
        ))
        
        # Iterate through all combinations
        combination_counter = 0
        
        for lora_idx, (r, lora_alpha, lora_dropout, target_modules) in enumerate(lora_combinations):
            logging.info(f"Processing LoRA combination {lora_idx + 1}/{len(lora_combinations)}")
            
            lora_config = {
                "r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules
            }
            
            for opt_name, opt_config in all_optimizers.items():
                logging.info(f"Processing optimizer: {opt_name}")
                
                # Generate combinations for this optimizer
                params, combinations = generate_optimizer_combinations(opt_config)
                
                for combo_idx, param_values in enumerate(combinations):
                    combination_counter += 1
                    
                    # Create parameter dictionary
                    training_params = {"optim": opt_config["optim"]}
                    for param, value in zip(params, param_values):
                        training_params[param] = value
                    
                    logging.info(f"Running combination {combination_counter}/{total_combinations}")
                    
                    # Run experiment
                    result = run_single_experiment(
                        lora_config, 
                        training_params, 
                        config, 
                        train_data, 
                        test_data_path, 
                        combination_counter
                    )
                    
                    # Save result immediately
                    save_results(result)
                    
                    if result["status"] == "success":
                        logging.info(f"Combination {combination_counter} completed successfully")
                    else:
                        logging.error(f"Combination {combination_counter} failed: {result.get('error', 'Unknown error')}")
        
        logging.info("All combinations completed successfully!")
        
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}")
        raise
    
    finally:
        # Final cleanup
        MemoryUtils.clear_cache()


if __name__ == "__main__":
    main()
