###
# train_lora_8bit.py: Definitive 8-bit LoRA training script with all
# compatibility fixes for k-bit training with gradient checkpointing on modern GPUs.
###

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
# Import all necessary functions from peft
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

# --- 1. Load Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- 2. Model and Data Paths ---
model_id = "meta-llama/Llama-2-7b-hf"
train_data_files = [
    "data/processed_train.json", "data/processed_tatqa_train.json", "data/processed_fiqa_train.json"
]
validation_data_files = [
    "data/processed_dev.json", "data/processed_tatqa_dev.json", "data/processed_fiqa_dev.json"
]

# --- 3. Load Model with Explicit 8-bit Quantization Config ---
print("Loading base model: meta-llama/Llama-2-7b-hf with 8-bit precision.")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    token=hf_token,
    device_map="auto"
)

# --- 4. Prepare Model for K-bit Training (CRUCIAL STEP) ---
print("Preparing 8-bit model for LoRA training...")
# This function handles several compatibility issues.
model = prepare_model_for_kbit_training(model)
print("Model prepared.")

# --- 5. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 6. Load and Tokenize Datasets ---
print("Loading and tokenizing datasets...")
train_dataset = load_dataset("json", data_files=train_data_files, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_files, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Successfully loaded and tokenized a combined dataset of {len(tokenized_train_dataset)} training samples.")


# --- 7. PEFT Configuration (LoRA) ---
# ### FIX 1: Target ALL linear layers in the Llama-2 model for stability ###
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA configured. Trainable parameters:")
model.print_trainable_parameters()

# --- 8. Training Arguments ---
# ### FIX 2: Explicitly set `use_reentrant=False` for gradient checkpointing ###
training_args = TrainingArguments(
    output_dir="./results_lora_8bit_combined/checkpoints",
    run_name="lora_8bit_combined_5epoch",
    report_to="wandb",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={'use_reentrant': False}, # Explicitly set this
    fp16=True,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# --- 9. Initialize Trainer ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 10. Start Training ---
print("\nStarting LoRA 8-bit training...")
trainer.train()
print("Training complete!")

# --- 11. Save the final best model ---
final_model_path = "./results_lora_8bit_combined/final_model"
trainer.save_model(final_model_path)
print(f"Final best model saved to {final_model_path}")

