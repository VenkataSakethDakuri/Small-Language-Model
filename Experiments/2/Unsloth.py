"""
Uses unsloth for lora training of gpt oss 20 b model. Has unsloth and vLLM inference. Uses a Math dataset from https://github.com/ziye2chen/DEMI-MathAnalysis 
for finetuning. Stores the output of finetuned model for llm as a judge evaluation.
"""

from turtle import pd
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from datasets import load_dataset, Dataset
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json


if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = None, # None for auto detection
   # max_seq_length = max_seq_length, 
   # load_in_4bit = True,  
    full_finetuning = False, 
)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
    
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

    trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

    trainer.train()

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

    # Save the model
    model.save_pretrained("adapter_gptOSSLora")
    tokenizer.save_pretrained("adapter_gptOSSLora")

    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "adapter_gptOSSLora",
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = False,
    )

    def get_model_size(model):
        params = sum(p.numel() for p in model.parameters())
        size_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return params, size_mb

    torch.cuda.reset_peak_memory_stats(device=0)

    param_count, model_size_mb = get_model_size(model)

    model.eval()

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

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True,
        reasoning_effort = "high",
    ).to(model.device)

    text_streamer = CustomTextStreamer(tokenizer)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens = 64, streamer = text_streamer)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    generated_outputs.append({
        'problem': row['Problem'],
        'original_solution': row['Solution'],
        'generated_solution': generated_text,
    })
        
torch.cuda.synchronize()
end_time = time.time()

with open("gpt_oss_Lora_unsloth.json", "w") as f:
    json.dump(generated_outputs, f)

print(f"\nInference Metrics:")
print(f"Model size: {model_size_mb:.2f} MB with {param_count} parameters")
print(f"Peak reserved memory: {round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)} GB")
print(f"Time to first token: {text_streamer.first_token_time - start_time:.5f} seconds")
print(f"Inference time: {end_time - start_time:.2f} seconds")

torch.cuda.empty_cache()