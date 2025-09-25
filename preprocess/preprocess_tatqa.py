###
# preprocess_tatqa.py
# This script preprocesses both the TRAIN and DEV sets of the raw TAT-QA dataset
# into the simple {"text": "..."} format, making them compatible with our existing scripts.
###

import os
import json
from tqdm import tqdm

# --- 1. SETUP: DEFINE FILE PATHS ---
print("Initializing script and defining paths...")

# Define paths for both train and dev sets
paths = {
    "train": {
        "input": "data/tatqa_dataset_train.json",
        "output": "data/processed_tatqa_train.json"
    },
    "dev": {
        "input": "data/tatqa_dataset_dev.json",
        "output": "data/processed_tatqa_dev.json"
    }
}

# --- 2. DEFINE DATA PROCESSING FUNCTIONS (Reused) ---

def format_table(table_data):
    """Formats the table dictionary into a readable string."""
    try:
        header = " | ".join(table_data.get('header', []))
        rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
        return f"TABLE:\n{header}\n{rows}"
    except Exception:
        return "TABLE:\n[Error formatting table]"

def create_prompts_for_tatqa_item(item):
    """
    Takes a single item from the TAT-QA dataset and yields a list of
    fully formatted text prompts, one for each question in the item.
    """
    paragraphs = "\n".join([p['text'] for p in item.get('paragraphs', [])])
    table_str = format_table(item.get('table', {}))
    context = f"{paragraphs}\n\n{table_str}".strip()

    prompts = []
    for qa_pair in item.get('questions', []):
        question = qa_pair.get('question', 'N/A')
        
        answer_data = qa_pair.get('answer', 'N/A')
        if isinstance(answer_data, list):
            answer = ", ".join(map(str, answer_data))
        else:
            answer = str(answer_data)

        final_text = f"""### INSTRUCTION:
Answer the question based on the context below.

### CONTEXT:
{context}

### QUESTION:
{question}

### ANSWER:
{answer}"""
        prompts.append({"text": final_text})
        
    return prompts

# --- 3. MAIN EXECUTION BLOCK ---

# Process both 'train' and 'dev' files
for split_name, split_paths in paths.items():
    input_path = split_paths["input"]
    output_path = split_paths["output"]

    if not os.path.exists(input_path):
        print(f"Input file not found for '{split_name}': {input_path}. Skipping.")
        continue

    print(f"\nProcessing '{split_name}' set...")
    print(f"Loading raw TAT-QA data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    print(f"Processing {len(raw_data)} entries from {split_name} set...")

    for item in tqdm(raw_data, desc=f"Preprocessing {split_name}"):
        prompts_for_item = create_prompts_for_tatqa_item(item)
        processed_data.extend(prompts_for_item)

    print(f"Generated a total of {len(processed_data)} processed data points for {split_name} set.")

    print(f"Saving processed {split_name} data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

print("\n--- TAT-QA preprocessing for all sets complete! ---")
