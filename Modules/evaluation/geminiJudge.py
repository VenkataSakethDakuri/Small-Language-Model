"""
LLM as a Judge evaluator (identical to llm_as_a_judge.py implementation).
"""

import time
import os
import json
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List
import multiprocessing

class Output(BaseModel):
    """Output model for LLM judge evaluation (identical to original)."""
    Problem: str
    first_score: int
    second_score: int

class LLMGeminiJudgeEvaluator:
    """LLM as a Judge evaluator using Gemini (identical to llm_as_a_judge.py)."""
    
    def __init__(self, base_model_name: str = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_model_name = base_model_name or self.config.get("base_model_name", "Qwen/Qwen2-0.5B")
        self.client = None
        self.base_model = None
        self.base_tokenizer = None
        self.setup_gemini()
        self.setup_base_model()
    
    def setup_gemini(self, **kwargs):
        """Setup Gemini client (identical to original implementation)."""
        load_dotenv()
        api_key = kwargs.get("api_key", self.config.get("gemini_api_key", os.getenv('GEMINI_API_KEY')))
        self.client = genai.Client(api_key=api_key)
    
    def setup_base_model(self, **kwargs):
        """Setup base model for comparison (identical to original implementation)."""
        model_name = kwargs.get("base_model_name", self.base_model_name)
        device_map = kwargs.get("device_map", self.config.get("device_map", "auto"))
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device_map
        )

    def setup_vllm_base_model(self, **kwargs):
        """Setup vLLM base model for comparison."""
        
        from vllm import LLM, SamplingParams
        
        model_name = kwargs.get("base_model_name", self.base_model_name)
        tensor_parallel_size = kwargs.get("tensor_parallel_size", self.config.get("tensor_parallel_size", 1))
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", self.config.get("gpu_memory_utilization", 0.8))
        
        # Initialize vLLM model
        self.vllm_base_model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        
        # Setup tokenizer for vLLM (needed for chat template)
        self.vllm_base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Store vLLM classes for later use
        self.SamplingParams = SamplingParams
    
    def compare_math_solutions(self, finetuned_output: str, base_output: str, 
                         problem: str, client, **kwargs) -> str:
        """Compare finetuned and base model solutions (identical to original implementation)."""
        model_name = kwargs.get("gemini_model", self.config.get("gemini_model", "gemini-2.5-pro"))
        
        prompt = f"""
        You are an expert in evaluating the solutions of math problems with respect to their logical completeness and formally justified solutions
        Compare these two solutions to the same problem and provide scoring.

        ### Problem:
        {problem}

        ### 1st Solution:
        {finetuned_output}

        ### 2nd Solution:
        {base_output}

        Evaluate both solutions on:
        1. Correctness (is the answer right?)
        2. Clarity and explanation quality
        3. Mathematical rigor
        4. Completeness

        Give the Final answer in this exact JSON format:
        
        {{
            "problem": "{problem}",
            "first_score": 5,
            "second_score": 5
        }}

        Replace the numbers with the actual values.
        """
        
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt,
            config={
            "response_mime_type": "application/json",
            "response_schema": Output,
        }
        )
        
        return response.text
    
    def generate_base_model_output(self, problem: str, **kwargs) -> str:
        """Generate output from base model (identical to original implementation)."""
        system_message = kwargs.get("system_message", self.config.get("system_message", 
            "Solve the following problem with logically complete and formally justified solutions:"))
        max_new_tokens = kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 128))
        
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user", 
                "content": f"### Problem:\n{problem}\n\n### Solution:"
            }
        ]

        prompt = self.base_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.base_tokenizer([prompt], return_tensors="pt").to("cuda")
        base_model_output = self.base_model.generate(**inputs, max_new_tokens=max_new_tokens)
        base_model_output = self.base_tokenizer.decode(base_model_output[0], skip_special_tokens=True)
        
        return base_model_output

    def generate_vLLM_base_model_output(self, problem: str, **kwargs) -> str:
        """Generate output from base model using vLLM."""
        # Initialize vLLM model if not already done
        if not hasattr(self, 'vllm_base_model') or self.vllm_base_model is None:
            self.setup_vllm_base_model(**kwargs)
        
        system_message = kwargs.get("system_message", self.config.get("system_message", "Solve the following problem with logically complete and formally justified solutions:"))
        max_tokens = kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 128))
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.0))
        top_p = kwargs.get("top_p", self.config.get("top_p", 1.0))
        
        # Prepare messages in chat format
        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user", 
                "content": f"### Problem:\n{problem}\n\n### Solution:"
            }
        ]
        
        # Apply chat template
        prompt = self.vllm_base_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Configure sampling parameters
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=kwargs.get("stop_tokens", None)
        )
        
        # Generate using vLLM
        outputs = self.vllm_base_model.generate([prompt], sampling_params)
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

    
    def evaluate_outputs(self, finetuned_outputs_file: str) -> List[Dict[str, Any]]:
        """Evaluate finetuned model outputs against base model (identical to original implementation)."""
        # Load finetuned outputs
        with open(finetuned_outputs_file, "r") as f:
            loaded_outputs = json.load(f)
        
        base_model_outputs = []
        comparison_results = []
        
        for output in loaded_outputs:
            # Generate base model output
            base_output = self.generate_base_model_output(output['problem'])
            base_model_outputs.append(base_output)
            
            # Compare solutions using Gemini
            if self.client:  # Only if Gemini client is properly set up
                comparison = self.compare_math_solutions(
                    output['generated_solution'],
                    base_output,
                    output['problem'],
                    self.client
                )
                comparison_results.append(comparison)
        
        return {
            "base_outputs": base_model_outputs,
            "comparisons": comparison_results
        }
    
    def calculate_average_scores(self, comparison_results: List[str]) -> Dict[str, float]:
        """Calculate average scores from comparison results."""
        total_first_score = 0
        total_second_score = 0
        valid_comparisons = 0
        
        for result in comparison_results:
            try:
                data = json.loads(result)
                total_first_score += data["first_score"]
                total_second_score += data["second_score"]
                valid_comparisons += 1
            except (json.JSONDecodeError, KeyError):
                continue
        
        if valid_comparisons == 0:
            return {"finetuned_avg": 0, "base_avg": 0}
        
        return {
            "finetuned_avg": total_first_score / valid_comparisons,
            "base_avg": total_second_score / valid_comparisons,
            "total_comparisons": valid_comparisons
        }

class LLMGeminiJudgeEvaluatorAsync:
    """Async wrapper for LLM Judge Evaluator with optimized batching."""
    
    def __init__(self, evaluator: LLMGeminiJudgeEvaluator, config: Dict[str, Any] = None):
        self.evaluator = evaluator
        self.config = config or {}
        
        # CPU-bound operations (base model generation)
        cpu_workers = self.config.get("cpu_workers", multiprocessing.cpu_count())
        self.cpu_executor = ThreadPoolExecutor(max_workers=cpu_workers)
        
        # I/O-bound operations (API calls)
        io_workers = self.config.get("io_workers", 25)  # Higher for I/O
        self.io_executor = ThreadPoolExecutor(max_workers=io_workers)
        
        # Batch sizes
        self.cpu_batch_size = self.config.get("cpu_batch_size", cpu_workers)
        self.io_batch_size = self.config.get("io_batch_size", min(io_workers, 20)) # Limit to avoid rate limits

    async def generate_base_model_output_async(self, problem: str, **kwargs) -> str:
        """Async wrapper for base model output generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor, 
            self.evaluator.generate_base_model_output, 
            problem, 
            **kwargs
        )
    
    async def generate_vLLM_base_model_output_async(self, problem: str, **kwargs) -> str:
        """Async wrapper for vLLM base model output generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.cpu_executor, 
            self.generate_vLLM_base_model_output, 
            problem, 
            **kwargs
        )


    async def compare_math_solutions_async(self, finetuned_output: str, base_output: str, 
                                         problem: str, **kwargs) -> str:
        """Async wrapper for Gemini API comparison."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.io_executor,
            self.evaluator.compare_math_solutions,
            finetuned_output,
            base_output,
            problem,
            self.evaluator.client,
            **kwargs
        )
    
    async def evaluate_outputs_async(self, finetuned_outputs_file: str) -> Dict[str, Any]:
        """Async evaluation of finetuned model outputs against base model."""
        # Load finetuned outputs
        with open(finetuned_outputs_file, "r") as f:
            loaded_outputs = json.load(f)
        
        base_model_outputs = []
        comparison_results = []
        
        # Step 1: Generate base model outputs in batches
        for i in range(0, len(loaded_outputs), self.cpu_batch_size):
            batch = loaded_outputs[i:i + self.cpu_batch_size]
            tasks = [
                self.generate_base_model_output_async(output['problem']) 
                for output in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            base_model_outputs.extend(batch_results)
        
        # Step 2: Compare solutions using Gemini in batches
        if self.evaluator.client:  # Only if Gemini client is properly set up
            for i in range(0, len(loaded_outputs), self.io_batch_size):
                batch = loaded_outputs[i:i + self.io_batch_size]
                tasks = [
                    self.compare_math_solutions_async(
                        output['generated_solution'],
                        base_model_outputs[i + idx],
                        output['problem']
                    ) 
                    for idx, output in enumerate(batch)
                ]
                batch_results = await asyncio.gather(*tasks)
                comparison_results.extend(batch_results)
        
        return {
            "base_outputs": base_model_outputs,
            "comparisons": comparison_results
        }

    async def calculate_average_scores_async(self, comparison_results: List[str]) -> Dict[str, float]:
        """Async calculation of average scores from comparison results."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.evaluator.calculate_average_scores,
            comparison_results
        )
    
    def cleanup(self):
        """Cleanup executors."""
        self.cpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.cleanup()



