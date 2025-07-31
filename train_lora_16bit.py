###
# train_lora_16bit.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig is no longer needed for 16-bit training
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv

# --- 1. Load Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- 2. Model and Data Paths ---
model_id = "meta-llama/Llama-2-7b-hf"
processed_data_path = "data/processed_train.json"

# --- 3. Load Model in 16-bit (FP16) Precision ---
print(f"Loading base model: {model_id} in 16-bit (FP16) precision.")

# We load the model in its default float16 precision.
# No quantization config is needed.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # Explicitly load in float16
    device_map="auto",
    token=hf_token
)
model.config.use_cache = False

# --- 4. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 5. Load and Tokenize Dataset ---
print(f"Loading and tokenizing dataset from: {processed_data_path}")
dataset = load_dataset("json", data_files=processed_data_path, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
print(f"Successfully loaded and tokenized {len(tokenized_dataset)} samples.")

# --- 6. PEFT Configuration (LoRA) ---

model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- 7. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results_16bit/checkpoints",
    run_name="16bit_15epoch_lr2e-5",
    report_to="wandb",
    num_train_epochs=15,
    per_device_train_batch_size=1, # Start with 1, can be increased on a 24GB GPU
    gradient_accumulation_steps=1,
    gradient_checkpointing=True, # Still useful for saving memory with activations
    learning_rate=2e-4,
    fp16=True, # Use fp16 for mixed-precision training
    logging_steps=500,
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
)

# --- 8. Initialize Trainer with Data Collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 9. Start Training ---
print("\nStarting 16-bit LoRA model training...")
trainer.train(resume_from_checkpoint=True)
print("Training complete!")

# --- 10. Save the final model ---
final_model_path = "./results_16bit/final_model"
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")
