"""
This code is an example of pipeline using standard transformers and PEFT libraries with LoRA adapters. 
It stores the adapter separately and later is used for inference. No quantization is used.
"""

from sympy import trunc
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    TextStreamer
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import load_dataset
import time
import psutil
import os

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    device_map="auto"
)

#GPT models (including DistilGPT-2) were originally designed for text generation, not batch processing. They don't include a dedicated padding token in their vocabulary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_alpha=16,
    lora_dropout=0, #dropout not used when inference_mode=True
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
    
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

dataset = dataset.map(data_processing, batched=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  
)

training_args = TrainingArguments(
    output_dir="TrainingRegular",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=108,
    report_to="none",
    save_strategy="epoch",
    dataloader_drop_last=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
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

torch.cuda.empty_cache()

print(f"\nTraining Metrics:")
print(f"{trainer.state.log_history[-1]['train_runtime'] if trainer.state.log_history else train_time_end - train_time_start:.2f} seconds used for training.")
print(
    f"{round((trainer.state.log_history[-1]['train_runtime'] if trainer.state.log_history else train_time_end - train_time_start)/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("Experiments/1/adapterRegular")
tokenizer.save_pretrained("Experiments/1/adapterRegular")

torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    device_map="auto"
)


model = PeftModel.from_pretrained(base_model, "Experiments/1/adapterRegular")
tokenizer = AutoTokenizer.from_pretrained("Experiments/1/adapterRegular")


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model.eval()

def get_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 * 1024)  
    return params, size_mb


param_count, model_size_mb = get_model_size(model)

start_time = time.time()

# Custom text streamer to measure time to first token
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
    ], return_tensors="pt").to("cuda")

text_streamer = CustomTextStreamer(tokenizer)

with torch.no_grad():
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

end_time = time.time()

print(f"\nInference Metrics:")
print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.2f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")

torch.cuda.empty_cache()