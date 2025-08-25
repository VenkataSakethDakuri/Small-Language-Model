"""
Code for using gemini model as LLM as a judge for evaluating responses from finetuned and base model.
"""
import time
from dotenv import load_dotenv
import os
from google import genai
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from pydantic import BaseModel

class Output(BaseModel):
    Problem: str
    first_score: int
    second_score: int

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

with open("gpt_oss_Lora.json", "r") as f:
    loaded_outputs = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="auto")

base_model_outputs = []
comparision_results = []


def compare_solutions(finetuned_output, base_output, problem, client):
    """Comparing the finetuned model output with the base model output."""
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
    comparision_result = response.text

    return comparision_result

for i in range(len(loaded_outputs)):

    messages = [
        {
            "role": "system",
            "content": "Solve the following problem with logically complete and formally justified solutions:"
        },
        {
            "role": "user", 
            "content": f"### Problem:\n{loaded_outputs[i]['problem']}\n\n### Solution:"
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    base_model_output = model.generate(**inputs, max_new_tokens=128)

    base_model_output = tokenizer.decode(base_model_output[0], skip_special_tokens=True)

    base_model_outputs.append({
        "problem": loaded_outputs[i]['problem'],
        "solution": base_model_output
    })

with open("base_model_outputs.json", "w") as f:
    json.dump(base_model_outputs, f)

# Now we have outputs from both the finetuned model and the base model. We can use Gemini to compare them.
client = genai.Client(api_key=GEMINI_API_KEY)

for i in range(len(loaded_outputs)):
    result = compare_solutions(loaded_outputs[i]['solution'], base_model_outputs[i]['solution'], loaded_outputs[i]['problem'], client)
    comparision_results.append(result)
    time.sleep(10) # To avoid rate limiting

with open("comparison_results.json", "w") as f:
    json.dump(comparision_results, f, indent=2)