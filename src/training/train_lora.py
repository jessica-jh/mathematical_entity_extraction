from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration for training
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "unsloth/Qwen2.5-Math-7B-Instruct" 
MAX_SEQ_LENGTH = 1024
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "math_lora_model")

# Prompt for training
alpaca_prompt = """### Instruction:
Extract all mathematical entities (definition, theorem, proof, example, name, reference) from the input. 

### Constraints:
- Output ONLY a valid JSON list of objects.
- DO NOT repeat the input, instruction, or add any conversational text.
- Each 'text' must be exactly as it appears in the input.

### Input:
{}

### Response:
{}""" 

def main():
    print("Initializing model and tokenizer")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )
    
    # LoRA setup
    model = FastLanguageModel.get_peft_model(
        model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407,
    )

    # Defining the formatting prompts function to make the tokenizer aware
    def formatting_prompts_func(examples):
        inputs  = examples["input"]
        outputs = examples["output"]
        texts = []
        for input_text, output_text in zip(inputs, outputs):
            # Adding EOS token to the output to make the model stop generating
            text = alpaca_prompt.format(input_text, output_text) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts }

    print(f"📦 Loading dataset from {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    training_args = TrainingArguments(
            output_dir = OUTPUT_DIR,
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            num_train_epochs = 15,          # Increased from 3 to 15
            warmup_steps = 10,
            learning_rate = 2e-4,
            lr_scheduler_type = "linear",
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,              # for real-time monitoring
            optim = "adamw_8bit",
            seed = 3407,
        )
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = training_args,
    )
    
    print("Starting training")
    trainer.train()
    
    final_model_path = os.path.join(OUTPUT_DIR, "final_lora_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Training completed. Saved to {final_model_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()