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

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    lora_alpha=16,
    lora_dropout=0, #dropout not used when inference_mode=True
    bias="none"
)

model = get_peft_model(model, lora_config)

# def data_processing(training_data):
#     instructions = training_data["instruction"]
#     inputs = training_data["input"]
#     outputs = training_data["output"]
#     texts = []

#     for instruction, input_text, output_text in zip(instructions, inputs, outputs):
#         text = f"### Instruction:\n{instruction}\n"
#         if input_text:
#             text += f"### Input:\n{input_text}\n"
#         text += f"### Response:\n{output_text}\n<|endoftext|>"
#         texts.append(text)
    
#     return {"text": texts}

def data_processing(training_data):
    instructions = training_data["instruction"]
    inputs = training_data["input"]
    outputs = training_data["output"]
    texts = []

    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        # Structure as a chat with system, user, and assistant
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n{input_text}" if input_text else instruction},
            {"role": "assistant", "content": output_text}
        ]
        # Apply Qwen2's chat template (assumes tokenizer has apply_chat_template)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:25000]")

dataset = dataset.map(data_processing, batched=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, #true for bert-like models, false for gpt-like models
    pad_to_multiple_of=8  
)

training_args = TrainingArguments(
    output_dir="TrainingRegular",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=5,
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

torch.cuda.reset_peak_memory_stats(device=0)

train_time_start = time.time()

trainer_result = trainer.train()

torch.cuda.synchronize()

train_time_end = time.time()

torch.cuda.empty_cache()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

torch.cuda.empty_cache()

print("Training Metrics:")
print(f"{train_time_start} - {train_time_end} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

model.save_pretrained("adapterRegular")
tokenizer.save_pretrained("adapterRegular")

torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "adapterRegular")
tokenizer = AutoTokenizer.from_pretrained("adapterRegular")

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
    size_mb = params * 4 / (1024 * 1024)  # No of bytes depends on the model type, here we assume float32
    return params, size_mb


param_count, model_size_mb = get_model_size(model)

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
            "Explain what is photosynthesis in simple terms", # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")

text_streamer = CustomTextStreamer(tokenizer)

torch.cuda.reset_peak_memory_stats(device=0)

start_time = time.time()

with torch.no_grad():
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

end_time = time.time()

print(f"\nInference Metrics:")
print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.5f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")

torch.cuda.empty_cache()