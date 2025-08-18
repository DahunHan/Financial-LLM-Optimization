# Rank Layers!
# Phase 2. Finally. Step 1 of the Layer Dropping Methodology.
# This script performs the Diagnosis phase to determine the importance of each layer in the model

import os
import json
import torch
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

# 1. SETUP : Load Environment Variables and define paths
# Load env variables from a .env file for security
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging face token loaded!")

# Define the base model and data paths
model_id = "meta-llama/Llama-2-7b-hf"
train_data_path = "data/processed_train.json"
output_dir = "./results_sapling/importance_probing"
ranking_output_path  = "layer_importance_ranking.json"

# 2. LOAD MODEL AND TOKENIZER
# Let us load the base Llama-2 model in 16-bit precision (bfloat16 for A100 compatibility)
print(f"Loading Base Model {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, #Use bfloat 16 for stability on A100 GPUs
    token=hf_token
)

# Load the tokenizer and configure it for training
# The padding token is set to the end-of-sentence token for causal language models
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code = True, token = hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Padding on the right is crucial for decoder-only models.

# 3. APPLY LORA TO ALL TRANSFORMER LAYERS
# The core of this diagnosis phase is to attach LoRA adapters to all layers
# This is to measure their individual response to fine-tuning.
print("Applying LoRA adapters to all target layers!")
lora_config = LoraConfig(
    r=16, # Rank of the update matrices. A higher rank means more parameters. 
    # We might want to change this rank in the future to see trend. (Obviously we want the highest per.)
    lora_alpha = 32, # A scaling factor for the LoRA updates.
    # Let us target all linear layers within the self-attention blocks to get a comprehensive importance score
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # This code makes sure that adapaters are attached at the 4 core linear layers.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
) 

# Use PEFT library to apply the LoRA config. to the model.
model = get_peft_model(model, lora_config)
print("LoRA adapters applied. Trainable parameters : ")
model.print_trainable_parameters() # This will show the number of parameters we are actually training

# 4. LOAD AND PREPARE DATASET
print(f"Loading and tokenizing training dataset from : {train_data_path}")
train_dataset = load_dataset("json", data_files=train_data_path, split="train")

# This function tokenizes the text data into a format the model can understand
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns = ["text"])
print ("Dataset successfully loaded and tokenized")

# 5. SHORT FINE-TUNING (1 EPOCH)
# We fine-tune for only one epoch. This is enough to probe the layers and see
# Which ones adapt the most to the new datam without fully converging.
print("\nStarting short fine-tuning (1 epoch) to probe layer importance...")
training_args=TrainingArguments(
    output_dir = output_dir,
    run_name="sapling_importance_probing",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=500,
    report_to="wandb",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)                            

# Start the training process.
trainer.train()
print("Short fine-tuning complete")

# 6. CALCULATE AND RANK LAYER IMPORTANCE
# After training, we analyze the trained LoRA weights to calculate importance scores
print("\nCalculating importance scores for each layer...")
layer_importance = defaultdict(float)

# Iterate through all the named parameters in the model.
for name, param in model.named_parameters():
    # We are only interested in the LoRA weights that were trained.
    if 'lora_' in name and param.requires_grad:
        # The layer index is extracted from the parameter name
        # i.e. "base_model.model.model.layers.15.self_....." -> 15
        try :
            layer_index = int(name.split('.')[3])
        except (ValueError, IndexError):
            continue # Skip if the name format is unexpected.
        # The importance score is the L2 Norm of the weight matrix.
        # A larger magnitude menas the layer's weight changed more during training
        # indicating it is more important for the task. 
        # Just think of it as defferentiating it twice to see the change of the trend.
        score = torch.norm(param.data.float(), p=2).item()
        layer_importance[layer_index] += score

# Convert the dictionary to a list of dictionaries for easier sorting and saving
ranked_layers = []
for index, score in layer_importance.items():
    ranked_layers.append({"layer_index": index,"importance_score": score})

# Sort the layers from LEAST important (lowerst score) to MOST important (highest score)
ranked_layers.sort(key = lambda x: x['importance_score'])

# 7. DISPLAY AND SAVE THE RANKING
print("\n--- Layer Importance Ranking (Least to Most Important)---")
for layer_info in ranked_layers:
    print(f"Layer {layer_info['layer_index']:<2}: Score = {layer_info['importance_score']:.4f}")
print("-------------------------------------------------------------")

# Save the detailed ranking to a JSON file for the next step (dropping)
with open(ranking_output_path, 'w') as f:
    json.dump(ranked_layers, f, indent=2)

print(f"\nLayer ranking has been saved to '{ranking_output_path}'.")
print("Phase 2, Step 1 (Diagnosis) is complete!")