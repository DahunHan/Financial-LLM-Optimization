###
# train_lora_8bit.py (with Gradient Checkpointing)
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
# Import the preparation function, as it's needed for gradient checkpointing with k-bit training
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

# --- Load Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- Model and Data Paths ---
model_id = "meta-llama/Llama-2-7b-hf"
processed_data_path = "data/processed_train.json"

# --- 1. Load Model with 8-bit Quantization ---
print(f"Loading base model: {model_id} with 8-bit precision.")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# --- ADDED: Prepare the model for k-bit training ---
# This is necessary for gradient checkpointing to work correctly with 8-bit models.
model = prepare_model_for_kbit_training(model)

# --- 2. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Load and Tokenize Dataset ---
print(f"Loading and tokenizing dataset from: {processed_data_path}")
dataset = load_dataset("json", data_files=processed_data_path, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
print(f"Successfully loaded and tokenized {len(tokenized_dataset)} samples.")

# --- 4. PEFT Configuration (LoRA) ---
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results_8bit",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    # --- ADDED: Enable gradient checkpointing ---
    # This will save a significant amount of memory during the training forward pass.
    gradient_checkpointing=True,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=25,
    remove_unused_columns=False,
)

# --- 6. Initialize Trainer with Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 7. Start Training ---
print("\nStarting model training (8-bit with gradient checkpointing)...")
trainer.train()
print("Training complete!")

# --- 8. Save the final model ---
final_model_path = "./results_8bit/final_model"
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")
