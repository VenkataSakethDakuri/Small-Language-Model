"""
This is pipeline for finetuning GPT-OSS-20b model using Lora. Uses a Math dataset from https://github.com/ziye2chen/DEMI-MathAnalysis 
for finetuning. Stores the output of finetuned model for llm as a judge evaluation.
"""
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
from datasets import Dataset
import time
import pandas as pd
import json

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", device_map="auto")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0, #dropout not used when inference_mode=True
    bias="none"
)

model = get_peft_model(model, lora_config)

def load_csv(file_path):
    return pd.read_csv(file_path)


#load training data
training_data = load_csv("pretraining_data.csv")

#preprocessing data
processed_data = []
for _, row in training_data.iterrows():
    Problem = row['Problem']
    Solution = row['Solution']
    
    messages = [
        {
            "role": "system",
            "content": "Solve mathematical problems with logically complete and formally justified solutions:"
        },
        {
            "role": "user",
            "content": f"### Problem:\n{row['Problem']}"
        },
        {
            "role": "assistant", 
            "content": f"### Solution:\n{row['Solution']}"
        }
    ]
#add generating prompt controls whether to add a prompt token that signals where the assistant should start responding, for training we want complete example so False, true in inference
#Tokenize here is false, later the trainer will automatically tokenize the input.
#add special tokens is false as chat template will anyways add the required tokens
    formatted_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, add_special_tokens=False)
    processed_data.append(formatted_text)

dataset = Dataset.from_dict({"text": processed_data})

def tokenize_dataset(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize_dataset, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, #true for bert-like models, false for gpt-like models
    pad_to_multiple_of=8  
)

training_args = TrainingArguments(
    output_dir="TrainingGPT_OSS",
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
    train_dataset=dataset,
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

print("Training Metrics:")
print(f"{train_time_end - train_time_start} seconds used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

model.save_pretrained("adapterGPT_OSS")
tokenizer.save_pretrained("adapterGPT_OSS")

torch.cuda.empty_cache()

base_model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "adapterGPT_OSS", device_map="auto") #not merging , creates a wrapper 
tokenizer = AutoTokenizer.from_pretrained("adapterGPT_OSS")

model.eval()

def get_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return params, size_mb

torch.cuda.reset_peak_memory_stats(device=0)

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

testing_dataset = pd.read_csv("benchmark_data.csv")

generated_outputs = []

start_time = time.time()

for i in range(len(testing_dataset)):
    row = testing_dataset.iloc[i]

    messages = [
        {
            "role": "system",
            "content": "Solve the following problem with logically complete and formally justified solutions:"
        },
        {
            "role": "user", 
            "content": f"### Problem:\n{row['Problem']}\n\n### Solution:"
        }
    ]
    
    # Apply chat template (this handles Harmony automatically)
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    text_streamer = CustomTextStreamer(tokenizer)

    torch.cuda.reset_peak_memory_stats(device=0)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, do_sample=False, temperature=0.5) #temperature ignored when do_sample is false

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True) #we need manual decoding if we want to store the generated text somewhere
    
    # Store results
    generated_outputs.append({
        'problem': row['Problem'],
        'original_solution': row['Solution'],
        'generated_solution': generated_text,
    })
        
torch.cuda.synchronize()
end_time = time.time()

with open("gpt_oss_Lora.json", "w") as f:
    json.dump(generated_outputs, f)

print(f"\nInference Metrics:")
print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.5f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")

torch.cuda.empty_cache()