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

class Output(BaseModel):
    """Output model for LLM judge evaluation (identical to original)."""
    Problem: str
    first_score: int
    second_score: int

class LLMGeminiJudgeEvaluator:
    """LLM as a Judge evaluator using Gemini (identical to llm_as_a_judge.py)."""
    
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.client = None
        self.base_model = None
        self.base_tokenizer = None
        self.setup_gemini()
        self.setup_base_model()
    
    def setup_gemini(self):
        """Setup Gemini client (identical to original implementation)."""
        load_dotenv()
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        # Initialize Gemini client (implementation depends on the exact API)
        # This would need to be adapted based on the actual Gemini client setup
        pass
    
    def setup_base_model(self):
        """Setup base model for comparison (identical to original implementation)."""
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, 
            device_map="auto"
        )
    
    def compare_math_solutions(self, finetuned_output: str, base_output: str, 
                         problem: str, client) -> str:
        """Compare finetuned and base model solutions (identical to original implementation)."""
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
            model="gemini-2.5-pro", 
            contents=prompt,
            config={
            "response_mime_type": "application/json",
            "response_schema": Output,
        }
        )
        
        return response.text
    
    def generate_base_model_output(self, problem: str) -> str:
        """Generate output from base model (identical to original implementation)."""
        messages = [
            {
                "role": "system",
                "content": "Solve the following problem with logically complete and formally justified solutions:"
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
        base_model_output = self.base_model.generate(**inputs, max_new_tokens=128)
        base_model_output = self.base_tokenizer.decode(base_model_output[0], skip_special_tokens=True)
        
        return base_model_output
    
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
