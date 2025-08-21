###
# prune_and_finetune.py
# Phase 2, Step 2 & 3 of the SAPLING methodology.
# This script performs the "Surgery" and "Rehabilitation" phases.
# 1. It reads the layer importance ranking.
# 2. It prunes (drops) a specified number of the least important layers.
# 3. It fine-tunes the new, smaller model to recover performance.
###

import os
import json
import torch
import argparse # To accept command-line arguments, like the number of layers to drop.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from dotenv import load_dotenv

# --- 0. ARGUMENT PARSER ---
# This allows us to run the script with different settings from the command line.
# For example: python prune_and_finetune.py --num_layers_to_drop 4
parser = argparse.ArgumentParser(description="Prune and fine-tune a Llama model based on layer importance.")
parser.add_argument(
    "--num_layers_to_drop",
    type=int,
    required=True,
    help="The number of least important layers to drop from the model."
)
args = parser.parse_args()


# --- 1. SETUP: LOAD ENVIRONMENT VARIABLES AND DEFINE PATHS ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# Dynamic paths based on the number of layers being dropped.
# This ensures that each experiment (dropping 4, 8, 12, etc.) has its own folder.
num_dropped = args.num_layers_to_drop
output_dir = f"./results_sapling/pruned_{num_dropped}_layers"

model_id = "meta-llama/Llama-2-7b-hf"
train_data_path = "data/processed_train.json"
validation_data_path = "data/processed_dev.json"
ranking_input_path = "layer_importance_ranking.json"


# --- 2. LOAD LAYER RANKING AND IDENTIFY LAYERS TO DROP ---
print(f"Loading layer importance ranking from: {ranking_input_path}")
with open(ranking_input_path, 'r') as f:
    ranked_layers = json.load(f)

# The list is sorted from least to most important, so we take the first N layers.
layers_to_drop_info = ranked_layers[:num_dropped]
layers_to_drop_indices = {layer['layer_index'] for layer in layers_to_drop_info}

print(f"Identified the {num_dropped} least important layers to drop: {sorted(list(layers_to_drop_indices))}")


# --- 3. LOAD BASE MODEL AND PRUNE LAYERS ("SURGERY") ---
print(f"Loading base model: {model_id} for pruning...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# The core pruning logic:
# We create a new list of layers, keeping only the ones we don't want to drop.
original_layers = base_model.model.layers
pruned_layers = [
    layer for i, layer in enumerate(original_layers) if i not in layers_to_drop_indices
]

# Replace the model's layer module list with our new, shorter list.
base_model.model.layers = torch.nn.ModuleList(pruned_layers)

# IMPORTANT: Update the model's configuration to reflect the new number of layers.
# This is crucial for the model to work correctly.
original_num_layers = base_model.config.num_hidden_layers
new_num_layers = len(pruned_layers)
base_model.config.num_hidden_layers = new_num_layers

print(f"\nPruning complete. Model layers reduced from {original_num_layers} to {new_num_layers}.")


# --- 4. FINE-TUNE THE PRUNED MODEL ("REHABILITATION") ---
# We can reuse the stable training setup from our full fine-tuning experiments.
print("Starting fine-tuning for the pruned model...")

pruned_model = base_model # Renaming for clarity
pruned_model.config.use_cache = False
pruned_model.gradient_checkpointing_enable()

# Load and tokenize datasets
train_dataset = load_dataset("json", data_files=train_data_path, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_path, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# Set up TrainingArguments for the pruned model.
# We'll train for 3 epochs as a starting point.
training_args = TrainingArguments(
    output_dir=f"{output_dir}/checkpoints",
    run_name=f"sapling_pruned_{num_dropped}_layers_ft",
    num_train_epochs=3, # Fine-tune for a few epochs to let the smaller model adapt.
    learning_rate=2e-5,
    optim="adamw_bnb_8bit", # Use the memory-efficient optimizer.

    # Memory Management
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    bf16=True,

    # Dataloader
    dataloader_num_workers=0,

    # Logging, Saving, and Evaluation
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
)

trainer = Trainer(
    model=pruned_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Start the fine-tuning process.
trainer.train()
print("Fine-tuning of the pruned model complete!")


# --- 5. SAVE THE FINAL COMPRESSED MODEL ---
final_model_path = f"{output_dir}/final_model"
trainer.save_model(final_model_path)
print(f"Final compressed and fine-tuned model saved to {final_model_path}")
print(f"Phase 2, Step 2 & 3 (Surgery & Rehabilitation) for dropping {num_dropped} layers is complete!")
