"""
This code is pipeline of making small language models using the LLM Compressor library with LoRA adapters. It stores the adapter separately
and later is used for inference. VLLM is used for inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextStreamer
from datasets import load_dataset
import time
import psutil
import os
from llmcompressor.transformers import oneshot
from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer (replacing FastLanguageModel.from_pretrained)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # Use 4-bit during training, reduces training memory not final model size
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Replace FastLanguageModel.get_peft_model with standard PEFT
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none"
)

model = get_peft_model(model, lora_config)

def data_processing(training_data):
    instructions = training_data["instruction"]
    inputs = training_data["input"]
    outputs = training_data["output"]
    texts = []

    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n"
        if input_text:
            text += f"### Input:\n{input_text}\n"
        text += f"### Response:\n{output_text}\n<|endoftext|>"
        texts.append(text)
    
    return {f"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:1000]")
dataset = dataset.map(data_processing, batched=True)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Replace SFTTrainer with standard Trainer
training_args = TrainingArguments(
    output_dir="outputsUnsloth",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=108,
    report_to="none",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Display current GPU stats to help monitor memory availability before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

train_time_start = time.time()


trainer_result = trainer.train()

train_time_end = time.time()

# Track final GPU memory and training time usage
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"\nTraining Metrics:")
print(f"{trainer_result.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_result.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Saving the adapter and base model seperately without merging, use gguf for merging
model.save_pretrained("Experiments/1/adapterLLMCompressor")
tokenizer.save_pretrained("Experiments/1/adapterLLMCompressor")

# Load the saved adapter (replacing FastLanguageModel.from_pretrained)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = get_peft_model(model, lora_config)
model.load_adapter("Experiments/1/adapterLLMCompressor")

merged_model = model.merge_and_unload() 

def create_calibration_dataset():
    calibration_samples = []
    for i in range(min(256, len(dataset))):
        sample = dataset[i]["text"]
        calibration_samples.append({"text": sample})
    
    return calibration_samples

calibration_dataset = create_calibration_dataset()

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=512, truncation=True, add_special_tokens=False)

calibration_dataset = dataset.map(tokenize, remove_columns=dataset.column_names, batched=True)

#need to play around with different recipes for quantization.
gtpq_recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(
        targets="Linear",
        scheme="W4A4",        #need to experiment with other schemes like W8A8, W4A8, etc.
        ignore=["lm_head"]
    )
]

oneshot(
    model=merged_model,
    dataset=calibration_dataset,
    recipe=gtpq_recipe,
    output_dir="Experiments/1/LLMCompressor"
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load the merged model for inference (replacing FastLanguageModel.for_inference)

model = merged_model
model.eval()

def get_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 * 1024)  # Assume 4 bytes per param
    return params, size_mb

# Measure model size
param_count, model_size_mb = get_model_size(model)

start_time = time.time()

#custom text streamer to measure time to first token
class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.first_token_time = 0
        self.flag = False
    
    def put(self, value):
        if not self.flag:
            self.first_token_time = time.time()
            self.flag = True
        
        super().put(value)

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = CustomTextStreamer(tokenizer)

with torch.no_grad():
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

end_time = time.time()

print(f"\nInference Metrics:")
print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.2f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")

#frees up unused GPU memory by emptying the CUDA memory cache
torch.cuda.empty_cache()

vllm_model = LLM(
    model="Experiments/1/UnslothLLMCompressor",
)

sampling_params = SamplingParams(
    temperature=0.5,
    max_tokens=512,
    stop=["<|endoftext|>"]
)

prompt = alpaca_prompt.format(
    "Continue the fibonnaci sequence.",
    "1, 1, 2, 3, 5, 8",
    ""
)

vllm_start_time = time.time()

response = vllm_model.generate([prompt], sampling_params=sampling_params)

vllm_end_time = time.time()

print(f"VLLM Inference time: {vllm_end_time - vllm_start_time:.2f} seconds")
print(f"VLLM Response: {response[0].outputs[0].text}")
