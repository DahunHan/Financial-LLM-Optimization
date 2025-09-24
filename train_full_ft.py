###
# train_full_ft.py: Final Full Fine-Tuning script using the combined
# FinQA, TAT-QA, and FiQA datasets.
###

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. Load Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- 2. Model and Data Paths (MODIFIED FOR FINAL EXPERIMENT) ---
model_id = "meta-llama/Llama-2-7b-hf"

# List all training files to be combined
train_data_files = [
    "data/processed_train.json",       # FinQA
    "data/processed_tatqa_train.json", # TAT-QA
    "data/processed_fiqa_train.json"   # FiQA
]

# List all validation files to be combined for in-training evaluation
validation_data_files = [
    "data/processed_dev.json",         # FinQA
    "data/processed_tatqa_dev.json",   # TAT-QA
    "data/processed_fiqa_dev.json"     # FiQA
]

# --- 3. Load Model for Full Fine-Tuning in BF16 Precision ---
print(f"Loading base model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
)

# --- 4. Configure Model for Training ---
model.config.use_cache = False
model.gradient_checkpointing_enable()

# --- 5. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 6. Load and Tokenize Datasets (MODIFIED TO LOAD MULTIPLE FILES) ---
print("Loading and tokenizing combined datasets...")

# The `load_dataset` function can take a list of files directly.
# The library will handle concatenating them into a single dataset.
train_dataset = load_dataset("json", data_files=train_data_files, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_files, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Successfully loaded and tokenized a combined dataset of {len(tokenized_train_dataset)} training samples.")

# --- 7. Training Arguments for Final Combined Training ---
training_args = TrainingArguments(
    # Directories and Naming (MODIFIED FOR FINAL EXPERIMENT)
    output_dir="./results_full_ft_combined/checkpoints",
    run_name="full_ft_combined_5epoch",
    report_to="wandb",

    # Training Hyperparameters
    num_train_epochs=5,
    learning_rate=2e-5,
    optim="adamw_bnb_8bit",

    # Memory Management
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    bf16=True,
    dataloader_num_workers=0,
    
    # Logging, Saving, and Evaluation
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# --- 8. Initialize Trainer ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 9. Start Training ---
print("\nStarting final Full Fine-Tuning on the combined dataset...")
trainer.train()
print("Training complete!")

# --- 10. Save the final best model ---
final_model_path = "./results_full_ft_combined/final_model"
trainer.save_model(final_model_path)
print(f"Final best model saved to {final_model_path}")