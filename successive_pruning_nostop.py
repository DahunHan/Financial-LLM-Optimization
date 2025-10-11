###
# successive_pruning_v3_final.py
# This is the definitive, corrected script for successive pruning.
# It correctly inherits the trained model from the previous iteration in memory,
# ensuring a true successive pruning process.
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

# --- 1. SETUP ---
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
ranking_input_path = "layer_importance_ranking.json"
base_output_dir = "./results_successive"

# --- 2. LOAD LAYER RANKING ---
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
print("Datasets successfully loaded.")

# --- 5. ITERATIVE PRUNING AND FINE-TUNING LOOP ---
results_log = []
best_model_so_far = None
lowest_eval_loss = float('inf')
# A map to track original layer indices to current indices
layer_index_map = list(range(current_model.config.num_hidden_layers))
max_layers_to_drop = 10 

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
        save_strategy="no", # We don't need to save checkpoints, we pass the model in memory
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
    results_log.append({'layers_remaining': current_layer_count, 'eval_loss': current_eval_loss})
    
    # Check for the best model but DO NOT stop the loop
    if current_eval_loss < lowest_eval_loss:
        lowest_eval_loss = current_eval_loss
        print(f"New best loss found! Saving this model state.")
        # Save the best model state to a specific directory
        best_model_path = f"{base_output_dir}/best_model_at_{current_layer_count}_layers"
        trainer.save_model(best_model_path)
    else:
        print(f"Eval loss of {current_eval_loss:.4f} is higher than the best loss of {lowest_eval_loss:.4f}, but continuing the run.")

    # --- CORRECTED & SIMPLIFIED: Pruning and Weight Inheritance Logic ---
    if i < max_layers_to_drop:
        # The trainer holds the fully trained model in memory. No need to reload from disk.
        current_model = trainer.model 
        
        original_index_to_drop = layers_to_drop_in_order[i]
        
        try:
            current_index_to_drop = layer_index_map.index(original_index_to_drop)
        except ValueError:
            print(f"Error: Could not find original layer index {original_index_to_drop}. Halting.")
            break

        print(f"\nPruning next layer for next iteration: Original Index {original_index_to_drop} (Current Position: {current_index_to_drop})")

        current_model.model.layers = torch.nn.ModuleList([
            layer for idx, layer in enumerate(current_model.model.layers) if idx != current_index_to_drop
        ])
        
        current_model.config.num_hidden_layers = len(current_model.model.layers)
        
        for new_idx, layer in enumerate(current_model.model.layers):
            layer.self_attn.layer_idx = new_idx
        
        layer_index_map.pop(current_index_to_drop)

# --- 6. LOG FINAL RESULTS ---
print("\n" + "="*60)
print("SUCCESSIVE PRUNING COMPLETE")
print("="*60)
print("Performance log:")
for result in results_log:
    print(f"  - Layers: {result['layers_remaining']}, Eval Loss: {result['eval_loss']:.4f}")

print(f"\nThe best performing model was saved at: {base_output_dir}/best_model_at_...")
print("Phase 2 (Successive Dropping) is complete!")