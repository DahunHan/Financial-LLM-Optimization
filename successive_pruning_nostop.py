###
# successive_pruning_v2_corrected.py
# This script implements a robust version of "Successive Dropping with Early Stopping".
# It correctly preserves learned weights between iterations and stops automatically
# when performance degrades, efficiently finding the optimal trade-off point.
###

import os
import json
import torch
import shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from dotenv import load_dotenv

# --- 1. SETUP: LOAD ENVIRONMENT VARIABLES AND DEFINE PATHS ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

model_id = "meta-llama/Llama-2-7b-hf"
train_data_files = [
    "data/processed_train.json", "data/processed_tatqa_train.json", "data/processed_fiqa_train.json"
]
# For this experiment, we use a combined validation set to get a reliable eval_loss
validation_data_files = [
    "data/processed_dev.json", "data/processed_tatqa_dev.json", "data/processed_fiqa_dev.json"
]
ranking_input_path = "layer_importance_ranking.json"
base_output_dir = "./results_successive"

# --- 2. LOAD LAYER RANKING ---
print(f"Loading layer importance ranking from: {ranking_input_path}")
with open(ranking_input_path, 'r') as f:
    ranked_layers = json.load(f)
layers_to_drop_in_order = [layer['layer_index'] for layer in ranked_layers]

# --- 3. LOAD INITIAL MODEL AND TOKENIZER ---
print(f"Loading initial base model: {model_id}")
current_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. LOAD AND TOKENIZE DATASETS ---
print("Loading and tokenizing datasets...")
train_dataset = load_dataset("json", data_files=train_data_files, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_files, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Datasets successfully loaded and tokenized.")

# --- 5. ITERATIVE PRUNING AND FINE-TUNING LOOP ---
lowest_eval_loss = float('inf')
best_model_checkpoint_path = None
best_model_layer_count = current_model.config.num_hidden_layers
# A map to track original layer indices to current indices
layer_index_map = list(range(current_model.config.num_hidden_layers))

# Let's check up to 10 layers dropped, as performance usually drops off steeply.
max_layers_to_drop = 10 

# First, run a baseline training on the full model (i=0)
for i in range(max_layers_to_drop + 1):
    num_dropped = i
    current_layer_count = current_model.config.num_hidden_layers
    
    print("\n" + "="*60)
    print(f"STARTING ITERATION {i}: DROPPED {num_dropped} LAYERS (Current layers: {current_layer_count})")
    print("="*60)

    iteration_output_dir = f"{base_output_dir}/run_{i}_drop_{num_dropped}"
    
    training_args = TrainingArguments(
        output_dir=iteration_output_dir,
        run_name=f"successive_drop_{num_dropped}",
        num_train_epochs=1,
        learning_rate=2e-5,
        optim="adamw_bnb_8bit",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True, # Let Trainer handle the best model in this small run
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model=current_model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    current_eval_loss = eval_metrics["eval_loss"]
    
    print(f"--- Iteration {i} Complete ---")
    print(f"Layers: {current_layer_count} -> Eval Loss: {current_eval_loss:.4f} (Best so far: {lowest_eval_loss:.4f})")
    
    # --- CORRECTED: Early Stopping Logic ---
    if current_eval_loss < lowest_eval_loss:
        lowest_eval_loss = current_eval_loss
        best_model_checkpoint_path = trainer.state.best_model_checkpoint # The path to the best model of this run
        best_model_layer_count = current_layer_count
        print(f"New best loss found! Storing checkpoint: {best_model_checkpoint_path}")
    else:
        print(f"Eval loss of {current_eval_loss:.4f} is higher than the best loss of {lowest_eval_loss:.4f}. Stopping.")
        break

    # --- CORRECTED: Pruning and Weight Inheritance Logic ---
    if i < max_layers_to_drop:
        # Load the best model from the completed run to prepare for the next iteration.
        print(f"\nLoading best model from {best_model_checkpoint_path} for next iteration...")
        current_model = AutoModelForCausalLM.from_pretrained(best_model_checkpoint_path, torch_dtype=torch.bfloat16)
        
        # Identify the next layer to drop based on the original ranking.
        original_index_to_drop = layers_to_drop_in_order[i]
        
        # Find the *current* position of that layer.
        # This is robust to previous drops.
        try:
            current_index_to_drop = layer_index_map.index(original_index_to_drop)
        except ValueError:
            print(f"Error: Could not find original layer index {original_index_to_drop} in current map. Skipping drop.")
            continue

        print(f"Pruning next layer: Original Index {original_index_to_drop} (Current Position: {current_index_to_drop})")

        # Prune the layer directly from the current model
        current_model.model.layers = torch.nn.ModuleList([
            layer for idx, layer in enumerate(current_model.model.layers) if idx != current_index_to_drop
        ])
        
        # Update the model's configuration and our index map
        current_model.config.num_hidden_layers = len(current_model.model.layers)
        layer_index_map.pop(current_index_to_drop)

# --- 6. SAVE THE BEST PERFORMING MODEL ---
if best_model_checkpoint_path:
    final_model_path = f"{base_output_dir}/final_best_model_{best_model_layer_count}_layers"
    print(f"\nSuccessive pruning finished. The best model had {best_model_layer_count} layers with a loss of {lowest_eval_loss:.4f}.")
    print(f"Copying best model from {best_model_checkpoint_path} to {final_model_path}")
    
    shutil.copytree(best_model_checkpoint_path, final_model_path, dirs_exist_ok=True)
    
    print("Best model saved successfully!")
else:
    print("\nNo training was successfully completed.")

print("Phase 2 (Successive Dropping) is complete!")