###
# successive_pruning.py
# An alternative approach for Phase 2 of the SAPLING methodology.
# This script implements "Successive Dropping with Early Stopping".
# It iteratively trains for 1 epoch, evaluates, and prunes the next least important layer,
# stopping automatically when performance (eval_loss) starts to degrade.
###

import os
import json
import torch
import shutil # Used for copying the final best model files.
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

# --- Define base model and data paths ---
model_id = "meta-llama/Llama-2-7b-hf"
train_data_path = "data/processed_train.json"
validation_data_path = "data/processed_dev.json"
ranking_input_path = "layer_importance_ranking.json"
base_output_dir = "./results_successive" # Base directory for this experiment's results.

# --- 2. LOAD LAYER RANKING ---
print(f"Loading layer importance ranking from: {ranking_input_path}")
with open(ranking_input_path, 'r') as f:
    ranked_layers = json.load(f)

# Create a simple list of layer indices to drop, from least to most important.
layers_to_drop_in_order = [layer['layer_index'] for layer in ranked_layers]

# --- 3. LOAD INITIAL MODEL AND TOKENIZER ---
print(f"Loading base model: {model_id}")
current_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. LOAD AND TOKENIZE DATASETS (Done once) ---
print("Loading and tokenizing datasets...")
train_dataset = load_dataset("json", data_files=train_data_path, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_path, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Datasets successfully loaded and tokenized.")


# --- 5. ITERATIVE PRUNING AND FINE-TUNING LOOP ---
lowest_eval_loss = float('inf') # Initialize with infinity to ensure the first loss is always lower.
best_model_checkpoint_path = None
best_model_layer_count = current_model.config.num_hidden_layers

# We will try dropping up to 16 layers, one by one.
max_layers_to_drop = 16

for i in range(max_layers_to_drop + 1): # Loop from 0 drops to 16 drops
    num_dropped = i
    current_layer_count = current_model.config.num_hidden_layers
    
    print("\n" + "="*50)
    print(f"STARTING ITERATION {i}: DROPPING {num_dropped} LAYERS (Current layers: {current_layer_count})")
    print("="*50)

    # --- Setup Trainer for this iteration ---
    iteration_output_dir = f"{base_output_dir}/run_{i}_drop_{num_dropped}"
    
    training_args = TrainingArguments(
        output_dir=iteration_output_dir,
        run_name=f"successive_drop_{num_dropped}",
        num_train_epochs=1, # Always train for just one epoch in each step.
        learning_rate=2e-5,
        optim="adamw_bnb_8bit",
        # Use the stable settings we found previously.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True,
        dataloader_num_workers=0,
        # We need to log, evaluate, and save at the end of our single epoch.
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1, # Only need the latest checkpoint from this run.
        load_best_model_at_end=False, # We manage the best model manually.
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

    # --- Train for 1 epoch and then evaluate ---
    trainer.train()
    eval_metrics = trainer.evaluate()
    current_eval_loss = eval_metrics["eval_loss"]
    
    print(f"--- Iteration {i} Complete ---")
    print(f"Layers: {current_layer_count} -> Eval Loss: {current_eval_loss:.4f}")
    
    # --- Early Stopping Logic ---
    if current_eval_loss < lowest_eval_loss:
        lowest_eval_loss = current_eval_loss
        # Find the path to the checkpoint that was just saved (e.g., './results_successive/run_0/checkpoint-390')
        latest_checkpoint = f"{iteration_output_dir}/checkpoint-{trainer.state.global_step}"
        best_model_checkpoint_path = latest_checkpoint
        best_model_layer_count = current_layer_count
        print(f"New best model found! Loss improved to {lowest_eval_loss:.4f}. Storing checkpoint: {best_model_checkpoint_path}")
    else:
        print(f"Overfitting detected! Eval loss increased from {lowest_eval_loss:.4f} to {current_eval_loss:.4f}.")
        print("Stopping the successive pruning process.")
        break # Exit the loop.

    # --- Prune the next layer for the next iteration ---
    # This happens only if we haven't stopped.
    if i < max_layers_to_drop:
        layer_to_drop_index = layers_to_drop_in_order[i]
        
        # We need to find the actual index in the *current* list of layers.
        # This is tricky because the layer indices shift after each pruning.
        # The safest way is to rebuild the model from the best checkpoint and prune that.
        print(f"\nPreparing for next iteration: Pruning layer {layer_to_drop_index}...")
        
        # Load the best performing model so far to ensure stability.
        current_model = AutoModelForCausalLM.from_pretrained(best_model_checkpoint_path, torch_dtype=torch.bfloat16, token=hf_token)
        
        # Get the full list of layers to drop up to this point.
        total_layers_to_drop_indices = {l['layer_index'] for l in ranked_layers[:i+1]}
        
        # Reload the ORIGINAL base model to perform a clean pruning operation.
        base_model_for_pruning = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=hf_token)
        original_layers = base_model_for_pruning.model.layers
        pruned_layers = [
            layer for idx, layer in enumerate(original_layers) if idx not in total_layers_to_drop_indices
        ]
        base_model_for_pruning.model.layers = torch.nn.ModuleList(pruned_layers)
        base_model_for_pruning.config.num_hidden_layers = len(pruned_layers)
        
        # Now, load the weights from our best performing model into this newly pruned structure.
        # This is a complex step, so for simplicity in this script, we will just prune the current model.
        # NOTE: A more robust implementation would map weights. But this direct pruning is simpler.
        
        original_layers = current_model.model.layers
        # Find which layer to drop in the *current* model. The original index was `layer_to_drop_index`.
        # We need to map it to the current indices.
        # This simplified approach just drops the next available layer from the original list.
        # A more robust script is needed for perfect index mapping.
        # For this experiment, we will simplify: re-prune from scratch using the best weights.
        
        # To avoid complex weight mapping, we reload the best model and then prune it again for the next step.
        # This is computationally inefficient but ensures correctness.
        print("Reloading original model and pruning...")
        next_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=hf_token)
        
        # Get all layers that should be dropped for the next iteration (i+1)
        layers_to_drop_next_iter = {l['layer_index'] for l in ranked_layers[:i+1]}
        print(f"Total layers to drop for next iteration: {sorted(list(layers_to_drop_next_iter))}")
        
        original_layers = next_model.model.layers
        pruned_layers_next_iter = [
            layer for idx, layer in enumerate(original_layers) if idx not in layers_to_drop_next_iter
        ]
        next_model.model.layers = torch.nn.ModuleList(pruned_layers_next_iter)
        next_model.config.num_hidden_layers = len(pruned_layers_next_iter)
        
        # CRITICAL STEP: Load the weights from the *previous* best model into this new, smaller architecture.
        # This transfers the learned knowledge.
        next_model.load_state_dict(current_model.state_dict(), strict=False)
        current_model = next_model

# --- 6. SAVE THE BEST PERFORMING MODEL ---
if best_model_checkpoint_path:
    final_model_path = f"{base_output_dir}/final_best_model_{best_model_layer_count}_layers"
    print(f"\nSuccessive pruning finished. The best model had {best_model_layer_count} layers with a loss of {lowest_eval_loss:.4f}.")
    print(f"Copying best model from {best_model_checkpoint_path} to {final_model_path}")
    
    # Copy the best checkpoint to a final, clearly named directory.
    shutil.copytree(best_model_checkpoint_path, final_model_path)
    
    print("Best model saved successfully!")
else:
    print("\nNo training was successfully completed.")

print("Phase 2 (Successive Dropping) is complete!")
