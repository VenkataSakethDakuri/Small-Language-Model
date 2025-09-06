"""
Simplified script for testing end-to-end training and evaluation pipeline.
Uses single parameter values to verify the workflow is working correctly.
Trains -> Standard Inference -> vLLM Inference -> Evaluation (async for both).
"""

import json
import logging
import sys
import os
import asyncio
from typing import Dict, Any
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.training.lora_trainer import LoRATrainer
from Modules.inference.standard_inference import StandardInferenceEngine
from Modules.inference.vllm_inference import VLLMInferenceEngine
from Modules.evaluation.geminiJudge import LLMGeminiJudgeEvaluator, LLMGeminiJudgeEvaluatorAsync
from Modules.utils.utlities import MemoryUtils, DataUtils, FileUtils

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_test.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_test_config() -> Dict[str, Any]:
    """Get simple test configuration with single values."""
    return {
        "model_name": "Qwen/Qwen3-1.7B",
        
        # LoRA parameters (single values)
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "o_proj"],
        
        # Training parameters (single values)
        "optim": "adamw_torch",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 2,
        "output_dir": "test_training_output"
    }

async def evaluate_standard_async(config: Dict[str, Any], inference_output_path: str) -> Dict[str, Any]:
    """Async evaluation of standard inference results (evaluation only)."""
    
    try:
        logging.info("Starting standard inference evaluation (async)")
        
        # Create base evaluator
        base_evaluator = LLMGeminiJudgeEvaluator(
            base_model_name=config["model_name"], 
            config={"model_name": config["model_name"]}
        )
        
        # Create async evaluator with optimized settings
        async_config = {
            "cpu_workers": 4,  # For base model generation
            "io_workers": 10,  # For Gemini API calls
            "cpu_batch_size": 2,  # Small batches for GPU memory
            "io_batch_size": 5   # Small batches to avoid rate limits
        }
        
        async with LLMGeminiJudgeEvaluatorAsync(base_evaluator, async_config) as async_evaluator:
            # Evaluate standard inference results
            logging.info("Processing standard inference results...")
            evaluation_results = await async_evaluator.evaluate_outputs_async(inference_output_path)
            average_scores = await async_evaluator.calculate_average_scores_async(evaluation_results["comparisons"])
            
            # Save intermediate comparison scores (append mode)
            standard_comparisons_path = "test_standard_comparisons.json"
            
            # Check if file exists and load existing data
            existing_data = {"experiments": []}
            if os.path.exists(standard_comparisons_path):
                try:
                    with open(standard_comparisons_path, "r") as f:
                        existing_data = json.load(f)
                        if "experiments" not in existing_data:
                            existing_data = {"experiments": [existing_data]}  # Convert old format
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = {"experiments": []}
            
            # Append new experiment data
            new_experiment = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "comparisons": evaluation_results["comparisons"],
                "metadata": {
                    "inference_output_file": inference_output_path,
                    "evaluation_type": "standard_inference",
                    "total_comparisons": len(evaluation_results["comparisons"])
                }
            }
            existing_data["experiments"].append(new_experiment)
            
            # Save updated data
            with open(standard_comparisons_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            logging.info(f"Standard inference comparison scores appended to {standard_comparisons_path} (experiment #{len(existing_data['experiments'])})")
            
            logging.info("Standard inference evaluation completed")
            
            return {
                "status": "success",
                "results": {
                    "finetuned_avg_score": average_scores["finetuned_avg"],
                    "base_avg_score": average_scores["base_avg"],
                    "total_comparisons": average_scores["total_comparisons"]
                },
                "evaluation_results": evaluation_results,
                "comparisons_file": standard_comparisons_path
            }
    
    except Exception as e:
        logging.error(f"Error in standard evaluation: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
    
    finally:
        # Clean up memory
        try:
            del base_evaluator
        except:
            pass
        import gc
        gc.collect()
        MemoryUtils.clear_cache()
        time.sleep(2)
        logging.info("Standard evaluation cleanup completed")

async def evaluate_vllm_async(config: Dict[str, Any], vllm_output_path: str) -> Dict[str, Any]:
    """Async evaluation of vLLM inference results (evaluation only)."""
    
    try:
        logging.info("Starting vLLM inference evaluation (async)")
        
        # Create base evaluator for vLLM evaluation
        vllm_base_evaluator = LLMGeminiJudgeEvaluator(
            base_model_name=config["model_name"], 
            config={"model_name": config["model_name"]}
        )
        
        # Create async evaluator for vLLM
        async_config = {
            "cpu_workers": 4,  # For base model generation
            "io_workers": 10,  # For Gemini API calls
            "cpu_batch_size": 2,  # Small batches for GPU memory
            "io_batch_size": 5   # Small batches to avoid rate limits
        }
        
        async with LLMGeminiJudgeEvaluatorAsync(vllm_base_evaluator, async_config) as vllm_async_evaluator:
            # Evaluate vLLM inference results
            logging.info("Processing vLLM inference results...")
            vllm_evaluation_results = await vllm_async_evaluator.evaluate_outputs_async(vllm_output_path)
            vllm_average_scores = await vllm_async_evaluator.calculate_average_scores_async(vllm_evaluation_results["comparisons"])
            
            # Save intermediate comparison scores for vLLM (append mode)
            vllm_comparisons_path = "test_vllm_comparisons.json"
            
            # Check if file exists and load existing data
            existing_data = {"experiments": []}
            if os.path.exists(vllm_comparisons_path):
                try:
                    with open(vllm_comparisons_path, "r") as f:
                        existing_data = json.load(f)
                        if "experiments" not in existing_data:
                            existing_data = {"experiments": [existing_data]}  # Convert old format
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = {"experiments": []}
            
            # Append new experiment data
            new_experiment = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "comparisons": vllm_evaluation_results["comparisons"],
                "metadata": {
                    "inference_output_file": vllm_output_path,
                    "evaluation_type": "vllm_inference",
                    "total_comparisons": len(vllm_evaluation_results["comparisons"])
                }
            }
            existing_data["experiments"].append(new_experiment)
            
            # Save updated data
            with open(vllm_comparisons_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            logging.info(f"vLLM inference comparison scores appended to {vllm_comparisons_path} (experiment #{len(existing_data['experiments'])})")
            
            logging.info("vLLM inference evaluation completed")
            
            return {
                "status": "success",
                "results": {
                    "finetuned_avg_score": vllm_average_scores["finetuned_avg"],
                    "base_avg_score": vllm_average_scores["base_avg"],
                    "total_comparisons": vllm_average_scores["total_comparisons"]
                },
                "evaluation_results": vllm_evaluation_results,
                "comparisons_file": vllm_comparisons_path
            }
    
    except Exception as e:
        logging.error(f"Error in vLLM evaluation: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }
    
    finally:
        # Clean up memory
        try:
            del vllm_base_evaluator
        except:
            pass
        MemoryUtils.synchronize()
        MemoryUtils.clear_cache()
        import gc
        gc.collect()
        time.sleep(2)
        logging.info("vLLM evaluation cleanup completed")

def run_test_experiment(config: Dict[str, Any], train_data: Any, test_data_path: str) -> Dict[str, Any]:
    """Run a single test training and evaluation experiment."""
    
    try:
        logging.info("Starting test training experiment")
        
        # Step 1: Training
        logging.info("=" * 50)
        logging.info("STEP 1: MODEL TRAINING")
        logging.info("=" * 50)
        
        # Create trainer instance
        trainer = LoRATrainer({"model_name": config["model_name"]})
        
        # Setup model with LoRA parameters
        trainer.setup_model(
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"]
        )
        
        # Prepare training parameters
        training_params = {
            "optim": config["optim"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "lr_scheduler_type": config["lr_scheduler_type"],
            "warmup_steps": config["warmup_steps"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "num_train_epochs": config["num_train_epochs"],
            "output_dir": config["output_dir"]
        }
        
        # Train the model
        logging.info("Starting model training")
        training_result = trainer.train(train_data, dataset_type="math", **training_params)
        logging.info(f"Training completed. Time: {training_result.get('training_time', 'N/A')}")
        
        # Clean up trainer
        del trainer
        MemoryUtils.synchronize()
        MemoryUtils.clear_cache()
        import gc
        gc.collect()
        time.sleep(2)
        
        # Step 2: Standard Inference
        logging.info("=" * 50)
        logging.info("STEP 2: STANDARD INFERENCE")
        logging.info("=" * 50)
        
        logging.info("Running standard inference")
        inference_engine = StandardInferenceEngine(config["output_dir"], {"model_name": config["model_name"]})
        inference_engine.load_model(base_model=config["model_name"], lora_path=config["output_dir"])
        
        inference_output_path = "test_inference_results.json"
        inference_results = inference_engine.batch_inference_on_math_dataset(test_data_path, inference_output_path)
        inference_engine.metrics()
        
        # Clean up standard inference engine
        del inference_engine
        MemoryUtils.synchronize()
        MemoryUtils.clear_cache()
        gc.collect()
        time.sleep(2)
        
        # Step 3: vLLM Inference
        logging.info("=" * 50)
        logging.info("STEP 3: vLLM INFERENCE")
        logging.info("=" * 50)
        
        logging.info("Running vLLM inference")
        vllm_engine = VLLMInferenceEngine(config["output_dir"], {"model_name": config["model_name"]})
        vllm_engine.load_model(base_model=config["model_name"], adapter_path=config["output_dir"])
        
        vllm_output_path = "test_vllm_results.json"
        vllm_results = vllm_engine.batch_inference_on_dataset(test_data_path, vllm_output_path)
        
        # Clean up vLLM engine
        del vllm_engine
        MemoryUtils.synchronize()
        MemoryUtils.clear_cache()
        gc.collect()
        time.sleep(2)
        
        # Step 4: Standard Evaluation
        logging.info("=" * 50)
        logging.info("STEP 4: STANDARD INFERENCE EVALUATION")
        logging.info("=" * 50)
        
        standard_result = asyncio.run(evaluate_standard_async(config, inference_output_path))
        
        if standard_result["status"] != "success":
            return {
                "status": "failed",
                "error": f"Standard evaluation failed: {standard_result.get('error', 'Unknown error')}",
                "training_results": {
                    "training_time": training_result.get("training_time", "N/A"),
                    "peak_memory": training_result.get("peak_memory", "N/A")
                }
            }
        
        # Memory cleanup between evaluations
        MemoryUtils.synchronize()
        MemoryUtils.clear_cache()
        gc.collect()
        time.sleep(3)
        
        # Step 5: vLLM Evaluation
        logging.info("=" * 50)
        logging.info("STEP 5: vLLM INFERENCE EVALUATION")
        logging.info("=" * 50)
        
        vllm_result = asyncio.run(evaluate_vllm_async(config, vllm_output_path))
        
        if vllm_result["status"] != "success":
            return {
                "status": "partial_success",
                "error": f"vLLM evaluation failed: {vllm_result.get('error', 'Unknown error')}",
                "training_results": {
                    "training_time": training_result.get("training_time", "N/A"),
                    "peak_memory": training_result.get("peak_memory", "N/A")
                },
                "standard_inference_results": standard_result["results"],
                "comparison_scores": {
                    "standard_comparisons_file": standard_result["comparisons_file"],
                    "vllm_comparisons_file": None
                }
            }
        
        # Combine all results
        final_result = {
            "status": "success",
            "lora_config": {
                "r": config["lora_r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": config["lora_dropout"],
                "target_modules": config["target_modules"]
            },
            "training_config": {
                "optimizer": config["optim"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["per_device_train_batch_size"],
                "epochs": config["num_train_epochs"]
            },
            "training_results": {
                "training_time": training_result.get("training_time", "N/A"),
                "peak_memory": training_result.get("peak_memory", "N/A")
            },
            "standard_inference_results": standard_result["results"],
            "vllm_inference_results": vllm_result["results"],
            "inference_outputs": {
                "standard_inference_file": inference_output_path,
                "vllm_inference_file": vllm_output_path
            },
            "comparison_scores": {
                "standard_comparisons_file": standard_result["comparisons_file"],
                "vllm_comparisons_file": vllm_result["comparisons_file"]
            }
        }
        
        logging.info("Test experiment completed successfully!")
        return final_result
        
    except Exception as e:
        logging.error(f"Error in test experiment: {str(e)}")
        MemoryUtils.clear_cache()
        return {
            "status": "failed",
            "error": str(e)
        }

def main():
    """Main function to run the test experiment."""
    start_time = time.time()
    # Setup logging
    setup_logging()
    logging.info("Starting test training and evaluation script")
    
    # Clear memory cache at start
    MemoryUtils.clear_cache()
    
    # Get test configuration
    config = get_test_config()
    
    # Data paths
    train_data_path = r"/home/interns/Workspaces/SLM_Experiments/Small-Language-Model/pretraining_data.csv"
    test_data_path = r"/home/interns/Workspaces/SLM_Experiments/Small-Language-Model/benchmark_data.csv"

    try:
        # Load training data
        logging.info("Loading training data")
        train_data = DataUtils.load_csv(train_data_path)
        logging.info(f"Training data loaded successfully")
        
        # Print configuration for verification
        logging.info("Test Configuration:")
        logging.info(f"  Model: {config['model_name']}")
        logging.info(f"  LoRA r: {config['lora_r']}, alpha: {config['lora_alpha']}")
        logging.info(f"  Optimizer: {config['optim']}")
        logging.info(f"  Learning rate: {config['learning_rate']}")
        logging.info(f"  Batch size: {config['per_device_train_batch_size']}")
        logging.info(f"  Epochs: {config['num_train_epochs']}")
        
        # Run test experiment
        result = run_test_experiment(config, train_data, test_data_path)
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        
        if result["status"] == "success":
            logging.info("✅ Test completed successfully!")
            logging.info("Results summary:")
            logging.info(f"  Training time: {result['training_results']['training_time']}")
            logging.info(f"  Standard inference score: {result['standard_inference_results']['finetuned_avg_score']:.3f}")
            logging.info(f"  vLLM inference score: {result['vllm_inference_results']['finetuned_avg_score']:.3f}")
            logging.info(f"  Standard inference file: {result['inference_outputs']['standard_inference_file']}")
            logging.info(f"  vLLM inference file: {result['inference_outputs']['vllm_inference_file']}")
            logging.info(f"  Standard comparison scores file: {result['comparison_scores']['standard_comparisons_file']}")
            logging.info(f"  vLLM comparison scores file: {result['comparison_scores']['vllm_comparisons_file']}")
        elif result["status"] == "partial_success":
            logging.warning("⚠️ Test partially completed!")
            logging.info("Results summary:")
            logging.info(f"  Training time: {result['training_results']['training_time']}")
            logging.info(f"  Standard inference score: {result['standard_inference_results']['finetuned_avg_score']:.3f}")
            logging.warning(f"  vLLM evaluation failed: {result.get('error', 'Unknown error')}")
            logging.info(f"  Standard comparison scores file: {result['comparison_scores']['standard_comparisons_file']}")
        else:
            logging.error(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        return 1
    
    finally:
        # Final cleanup
        MemoryUtils.clear_cache()
        logging.info("Test script completed")
    
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Total script time: {total_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
