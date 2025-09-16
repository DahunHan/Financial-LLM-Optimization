###
# train_full_ft.py: Full Fine-Tuning script for Llama-2-7B on A100
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from dotenv import load_dotenv

# --- 1. Load Hugging Face Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- 2. Model and Data Paths ---
model_id = "meta-llama/Llama-2-7b-hf"
train_data_path = "data/augmented_train.json"
validation_data_path = "data/processed_dev.json"

# --- 3. Load Model for Full Fine-Tuning in BF16 Precision ---
print(f"Loading base model: {model_id} for Full Fine-Tuning in BF16 precision.")
# A100 GPUs support bfloat16 for more stable training
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
)

# --- 4. Configure Model for Training ---
# Disable cache for training, enable gradient checkpointing for memory efficiency
model.config.use_cache = False
model.gradient_checkpointing_enable()

# --- 5. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 6. Load and Tokenize Datasets (Robust Method) ---
import json
from datasets import Dataset
from tqdm import tqdm

print("Loading and manually processing datasets...")

def create_prompt_for_item(data_item):
    """
    Takes a raw data item from FinQA json and formats it into the final text prompt.
    This is similar to the logic in preprocess_data.py.
    """
    table_list = data_item.get('table', [])
    table_str = ""
    if table_list:
        # Handle potential nesting
        table_content = table_list[0] if isinstance(table_list[0], list) else table_list
        table_data = table_content[0] if isinstance(table_content, list) and table_content else {}

        if isinstance(table_data, dict):
            header = " | ".join(table_data.get('header', []))
            rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
            table_str = f"TABLE:\n{header}\n{rows}"
        elif isinstance(table_data, str):
             table_str = f"TABLE:\n{table_data}"

    pre_text = "\n".join(data_item.get('pre_text', []))
    post_text = "\n".join(data_item.get('post_text', []))

    qa_data = data_item.get('qa', {})
    question = qa_data.get('question', 'N/A')
    answer = qa_data.get('answer', 'N/A')

    # The final text format that the model will be trained on.
    return f"""### INSTRUCTION:
    Answer the question based on the context below.

    CONTEXT:
    {pre_text}
    {table_str}
    {post_text}

    QUESTION:
    {question}

    ANSWER:
    {answer}"""

# Manually load the JSON files.
with open(train_data_path, 'r', encoding='utf-8') as f:
    train_json_data = json.load(f)
with open(validation_data_path, 'r', encoding='utf-8') as f:
    validation_json_data = json.load(f)

# Process each item into the {"text": "..."} format.
train_processed = [{"text": create_prompt_for_item(item)} for item in tqdm(train_json_data, desc="Processing train data")]
validation_processed = [{"text": create_prompt_for_item(item)} for item in tqdm(validation_json_data, desc="Processing validation data")]

# Create Dataset objects from the processed lists.
train_dataset = Dataset.from_dict({"text": [item["text"] for item in train_processed]})
validation_dataset = Dataset.from_dict({"text": [item["text"] for item in validation_processed]})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_validation_dataset = validation_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Successfully loaded and tokenized datasets.")

# --- 7. Training Arguments for Full Fine-Tuning ---
training_args = TrainingArguments(
    # Directories and Naming
    output_dir="./results_full_ft_augmented/checkpoints",
    run_name="full_ft_augmented_5epoch_lr2e-5",
    report_to="wandb",

    # Training Hyperparameters
    num_train_epochs=5, # As requested: 5 epochs
    learning_rate=2e-5,
    # Memory efficient 8-bit optimizer
    optim="adamw_bnb_8bit",
    # Memory Management for Full Fine-Tuning on A100
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16, # Effective batch size = 1 * 16 = 16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    bf16=True, # Use bfloat16 for stable training on Ampere GPUs

    ## Disable dataloader multiprocessing
    dataloader_num_workers=0,
    
    # Logging, Saving, and Evaluation
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3, # Save only the last 3 checkpoints
    load_best_model_at_end=True, # Load the best model based on eval_loss
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
print("\nStarting Full Fine-Tuning...")
trainer.train()
print("Training complete!")

# --- 10. Save the final best model ---
final_model_path = "./results_full_ft/final_model"
trainer.save_model(final_model_path)
print(f"Final best model saved to {final_model_path}")