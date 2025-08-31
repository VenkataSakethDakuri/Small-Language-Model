"""
Script for end to end training and evaluation of a model with math dataset using gemini LLM as a judge. Uses standard and vLLM inference. 
Uses a range of values and options for hyperparameters for LoRA training
"""

import itertools
import json
from Modules.training.lora_trainer import LoRATrainer
from Modules.inference.standard_inference import StandardInferenceEngine
from Modules.inference.vllm_inference import VLLMInferenceEngine
from Modules.evaluation.geminiJudge import LLMGeminiJudgeEvaluator
from Modules.utils.utlities import MemoryUtils, DataUtils, FileUtils

MemoryUtils.clear_cache()

lora_params = {
    # LoRA rank - controls adaptation capacity
    "r": [4, 8, 16, 32, 64],  # Current: 8
    
    # LoRA alpha - controls scaling of adapter weights  
    "lora_alpha": [8, 16, 32, 64, 128],  # Current: 16
    # Common ratios: alpha = r, alpha = 2*r, or alpha = r/2
    
    # LoRA dropout - regularization for adapters
    "lora_dropout": [0.0, 0.05, 0.1, 0.2],  # Current: 0
    
    # Target modules - which layers to adapt
    "target_modules": [
        # Conservative (faster, less memory)
        ["q_proj", "v_proj"],
        
        # Balanced (your current setup)  
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Aggressive (more parameters)
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]
    ],
    
    # Bias handling - keep as "none" for most cases
    "bias": "none"  # Fixed - don't change
}

# Create trainer configuration
config = {
    "model_name": "openai/gpt-oss-20b"
}

# OPTION 1: Direct replacements - same parameter ranges
direct_replacements = {
    # Your current setup - baseline
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
    
    # Best direct upgrade - same parameter ranges
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
    
    # Improved 8-bit version
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
    
    # Similar requirements to Adam
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

# OPTION 2: Optimizers Requiring Parameter Adjustments (Full Parameter Sets)
adjusted_optimizers = {
    # SGD - Requires higher LR and benefits from scheduler
    "sgd": {
        "optim": "sgd",
        "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2],  # 5-10x higher than Adam
        "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
        "lr_scheduler_type": ["cosine", "linear", "polynomial", "constant_with_warmup"],  # Cosine highly recommended
        "warmup_steps": [5, 10, 25, 50, 100, 200],               # Often needs more warmup
        "per_device_train_batch_size": [1, 2, 4, 8, 16],         # Same ranges
        "gradient_accumulation_steps": [2, 4, 8, 16, 32],        # May benefit from larger effective batch
        "num_train_epochs": [2, 3, 5, 8, 10],                    # Often needs more epochs
    },
    
    # Lion - Requires much lower LR, handles higher weight decay
    "lion": {
        "optim": "lion",
        "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4], # 2-4x lower than Adam
        "weight_decay": [0.01, 0.05, 0.1, 0.2, 0.3],            # Can handle higher values
        "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
        "warmup_steps": [0, 5, 10, 25, 50, 100],
        "per_device_train_batch_size": [1, 2, 4, 8, 16],
        "gradient_accumulation_steps": [1, 2, 4, 8, 16],
        "num_train_epochs": [1, 2, 3, 5, 8],
    },
    
    # Adafactor - Ignores LR, has built-in regularization
    "adafactor": {
        "optim": "adafactor",
        "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],             # Often ignored by optimizer
        "weight_decay": [0.0],                                  # Fixed - use 0.0 (built-in regularization)
        "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup"],
        "warmup_steps": [0, 5, 10, 25, 50],                    # Less sensitive to warmup
        "per_device_train_batch_size": [1, 2, 4, 8, 16],
        "gradient_accumulation_steps": [1, 2, 4, 8, 16],
        "num_train_epochs": [1, 2, 3, 5, 8],
    },
    
    # Adagrad - For sparse gradients
    "adagrad": {
        "optim": "adagrad",
        "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],       # Higher LR needed
        "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
        "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
        "warmup_steps": [0, 5, 10, 25, 50, 100],
        "per_device_train_batch_size": [1, 2, 4, 8, 16],
        "gradient_accumulation_steps": [1, 2, 4, 8, 16],
        "num_train_epochs": [1, 2, 3, 5, 8],
    }
}

# OPTION 3: Advanced/Experimental Optimizers (Full Parameter Sets)
advanced_optimizers = {
    # Fused AdamW (if available)
    "adamw_apex_fused": {
        "optim": "adamw_apex_fused",
        "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],      # Same as standard AdamW
        "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
        "lr_scheduler_type": ["linear", "cosine", "constant", "constant_with_warmup", "polynomial"],
        "warmup_steps": [0, 5, 10, 25, 50, 100],
        "per_device_train_batch_size": [1, 2, 4, 8, 16],
        "gradient_accumulation_steps": [1, 2, 4, 8, 16],
        "num_train_epochs": [1, 2, 3, 5, 8],
    },
    
    # AdamW HF variant
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
    
    # Lamb optimizer (for large batch training)
    "lamb": {
        "optim": "lamb",
        "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],       # Similar to AdamW but can go higher
        "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
        "lr_scheduler_type": ["linear", "cosine", "polynomial", "constant_with_warmup"],
        "warmup_steps": [5, 10, 25, 50, 100, 200],              # Often benefits from warmup
        "per_device_train_batch_size": [2, 4, 8, 16, 32],       # Better with larger batches
        "gradient_accumulation_steps": [1, 2, 4, 8, 16],
        "num_train_epochs": [1, 2, 3, 5, 8],
    }
}

# Combine all optimizer configurations
all_optimizers = {**direct_replacements, **adjusted_optimizers, **advanced_optimizers}

# Generate all combinations of LoRA parameters
lora_combinations = list(itertools.product(
    lora_params["r"],
    lora_params["lora_alpha"], 
    lora_params["lora_dropout"],
    lora_params["target_modules"]
))

# Function to generate optimizer parameter combinations
def generate_optimizer_combinations(optimizer_config):
    """Generate all combinations for a single optimizer configuration."""
    params = []
    values = []
    
    for param, value_list in optimizer_config.items():
        if param != "optim":  # Skip the optimizer name
            params.append(param)
            values.append(value_list)
    
    combinations = list(itertools.product(*values))
    return params, combinations

# Calculate total combinations
total_combinations = 0
optimizer_counts = {}

for opt_name, opt_config in all_optimizers.items():
    params, combinations = generate_optimizer_combinations(opt_config)
    optimizer_counts[opt_name] = len(combinations)
    total_combinations += len(combinations)

total_with_lora = total_combinations * len(lora_combinations)

# Load training data once
train_data = DataUtils.load_csv(r"C:\Users\DELL\OneDrive\Desktop\Project\SLM\Experiments\2\pretraining_data.csv")

# Iterate through all combinations
results = []
combination_counter = 0

for lora_idx, (r, lora_alpha, lora_dropout, target_modules) in enumerate(lora_combinations):
    
    for opt_name, opt_config in all_optimizers.items():
        
        # Generate combinations for this optimizer
        params, combinations = generate_optimizer_combinations(opt_config)
        
        for combo_idx, param_values in enumerate(combinations):
            combination_counter += 1
            
            # Create parameter dictionary
            training_params = {"optim": opt_config["optim"]}
            for param, value in zip(params, param_values):
                training_params[param] = value
            
            # Create trainer instance
            trainer = LoRATrainer(config)
            
            # Setup model with current LoRA parameters
            trainer.setup_model(
                lora_r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            
            # Create unique output directory
            output_dir = f"TrainingMath_LoRA"
            training_params["output_dir"] = output_dir
            
            # Train the model
            training_result = trainer.train_math(train_data, **training_params)
            
            # Run standard inference
            inference_engine = StandardInferenceEngine(output_dir, config)
            inference_engine.load_model(base_model=config["model_name"], lora_path=output_dir)
            
            # Run inference on test dataset
            test_data_path = r"C:\Users\DELL\OneDrive\Desktop\Project\SLM\Experiments\2\benchmark_data.csv"
            inference_output_path = f"inference_results.json"
            inference_results = inference_engine.batch_inference_on_math_dataset(test_data_path, inference_output_path)
            
            # Evaluate standard inference results
            evaluator = LLMGeminiJudgeEvaluator(base_model_name=config["model_name"], config=config)
            evaluation_results = evaluator.evaluate_outputs(inference_output_path)
            average_scores = evaluator.calculate_average_scores(evaluation_results["comparisons"])

            MemoryUtils.clear_cache()
            
            # Run vLLM inference
            vllm_engine = VLLMInferenceEngine(output_dir, config)
            vllm_engine.load_model(base_model=config["model_name"], adapter_path=output_dir)
            
            # Run vLLM inference on test dataset
            vllm_output_path = f"vllm_inference_results.json"
            vllm_results = vllm_engine.batch_inference_on_dataset(test_data_path, vllm_output_path)
            
            # Evaluate vLLM inference results
            vllm_evaluator = LLMGeminiJudgeEvaluator(base_model_name=config["model_name"], config=config)
            vllm_evaluation_results = vllm_evaluator.evaluate_outputs(vllm_output_path)
            vllm_average_scores = vllm_evaluator.calculate_average_scores(vllm_evaluation_results["comparisons"])
            MemoryUtils.clear_cache()

            # Store result
            result = {
                "combination_id": combination_counter,
                "lora_config": {
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "target_modules": target_modules
                },
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
            
            # Load existing results if file exists
            try:
                with open("hyperparameter_results.json", "r") as f:
                    all_results = json.load(f)
            except FileNotFoundError:
                all_results = []
            
            # Add current result
            all_results.append(result)
            
            # Save updated results
            with open("hyperparameter_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            vllm_engine.cleanup()
            MemoryUtils.clear_cache()
    
# Final summary and save
try:
    with open("hyperparameter_results.json", "r") as f:
        all_results = json.load(f)
except FileNotFoundError:
    all_results = []



            













