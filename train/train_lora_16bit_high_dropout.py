###
# train_lora_16bit_high_dropout.py
# This script fine-tunes the Llama-2-7B model using 16-bit LoRA with an increased
# dropout rate to enhance regularization and combat overfitting.
###

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

# --- 1. SETUP: LOAD TOKEN AND DEFINE PATHS ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

model_id = "meta-llama/Llama-2-7b-hf"
train_data_files = [
    "data/processed_train.json", "data/processed_tatqa_train.json", "data/processed_fiqa_train.json"
]
validation_data_files = [
    "data/processed_dev.json", "data/processed_tatqa_dev.json", "data/processed_fiqa_dev.json"
]

# --- 2. LOAD 16-BIT MODEL AND TOKENIZER ---
print(f"Loading base model: {model_id} in BF16 precision.")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for A100 GPUs
    token=hf_token,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. LOAD AND TOKENIZE DATASETS ---
print("Loading and tokenizing datasets...")
train_dataset = load_dataset("json", data_files=train_data_files, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_files, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Datasets successfully loaded.")

# --- 4. PEFT CONFIGURATION (LoRA with High Dropout) ---
print("Configuring LoRA with a high dropout rate for strong regularization.")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    # --- KEY CHANGE: Increased dropout rate ---
    lora_dropout=0.25,
    # -----------------------------------------
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA configured. Trainable parameters:")
model.print_trainable_parameters()

# --- 5. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./results_lora_16bit_dropout/checkpoints",
    run_name="lora_16bit_high_dropout_0.25",
    report_to="wandb",
    num_train_epochs=2, # Train for 2 epochs to observe the effect
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True, # Use bfloat16
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# --- 6. INITIALIZE TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# --- 7. START TRAINING ---
print("\nStarting LoRA 16-bit training with high dropout...")
trainer.train()
print("Training complete!")

# --- 8. SAVE THE FINAL BEST MODEL ---
final_model_path = "./results_lora_16bit_dropout/final_model"
trainer.save_model(final_model_path)
print(f"Final best model saved to {final_model_path}")