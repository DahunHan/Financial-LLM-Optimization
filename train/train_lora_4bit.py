###
# train_qlora_4bit_with_monitoring.py
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

# --- 3. Load Model with 4-bit QLoRA Quantization ---
print(f"Loading base model: {model_id} with 4-bit QLoRA precision.")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)

# --- 4. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 6. Load and Tokenize Datasets ---
print("Loading and tokenizing datasets...")
train_dataset = load_dataset("json", data_files=train_data_files, split="train")
validation_dataset = load_dataset("json", data_files=validation_data_files, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
print(f"Successfully loaded and tokenized datasets.")

# --- 6. PEFT Configuration (LoRA) ---
lora_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- 7. Training Arguments (MODIFIED FOR MONITORING) ---
training_args = TrainingArguments(
    output_dir="./results_4bit_combined/checkpoints",
    run_name="4bit_15epoch_lr2e-5",
    report_to="wandb",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    remove_unused_columns=False,
)

# --- 8. Initialize Trainer with Data Collator and Eval Dataset ---
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
print("\nStarting model training with monitoring...")
trainer.train()
print("Training complete!")

# --- 10. Save the final model ---
final_model_path = "./results_4bit_combined/final_model"
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")