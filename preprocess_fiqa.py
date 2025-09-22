###
# preprocess_fiqa.py
# This script preprocesses the FiQA dataset into the simple {"text": "..."} format.
# This definitive version correctly parses the actual FiQA data structure, which is
# a list of dictionaries with "instruction", "input", and "output" keys.
###

import os
import json
from tqdm import tqdm

# --- 1. SETUP: DEFINE FILE PATHS ---
print("Initializing script and defining paths...")
paths = {
    "train": {
        "input": "data/fiqa_train.json",
        "output": "data/processed_fiqa_train.json"
    },
    "dev": {
        "input": "data/fiqa_test.json",
        "output": "data/processed_fiqa_dev.json"
    }
}

# --- 2. DEFINE THE FINAL, CORRECT DATA PROCESSING FUNCTION ---

def create_prompt_for_fiqa_item(item):
    """
    Takes a single dictionary item from the FiQA dataset and formats it
    into the standard text prompt format.
    """
    # The keys in the FiQA dataset are 'instruction', 'input', 'output'.
    instruction = item.get("instruction")
    question = item.get("input")
    answer = item.get("output")

    # Final validation: Ensure we have the critical pieces of information.
    if not all([instruction, question, answer]):
        return None

    # For FiQA, the 'instruction' itself serves as the main context.
    context = "CONTEXT:\n" + instruction.strip()

    # The final text format that the model will be trained on.
    final_text = f"""### INSTRUCTION:
Answer the question based on the context below. The question is about financial opinions, sentiment, or cause-and-effect.

{context}

### QUESTION:
{question}

### ANSWER:
{answer}"""
    return {"text": final_text}

# --- 3. MAIN EXECUTION BLOCK ---

# Process both 'train' and 'dev' (from test) files
for split_name, split_paths in paths.items():
    input_path = split_paths["input"]
    output_path = split_paths["output"]

    if not os.path.exists(input_path):
        print(f"Input file not found for '{split_name}': {input_path}. Skipping.")
        continue

    print(f"\nProcessing '{split_name}' set...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    print(f"Processing {len(raw_data)} entries from {split_name} set...")

    for item in tqdm(raw_data, desc=f"Preprocessing {split_name}"):
        processed_item = create_prompt_for_fiqa_item(item)
        if processed_item:
            processed_data.append(processed_item)

    # Final check to ensure data was processed
    if not processed_data:
        print(f"Warning: No data was processed for the '{split_name}' set. The output file will be empty. Check the JSON structure.")
    else:
        print(f"Generated a total of {len(processed_data)} processed data points for {split_name} set.")

    print(f"Saving processed {split_name} data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

print("\n--- FiQA preprocessing for all sets complete! ---")

