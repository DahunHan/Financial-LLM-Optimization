###
# combine_dev_datasets.py
# This script combines the validation/development sets from FinQA, TAT-QA, and FiQA
# into a single, unified validation set with a consistent format.
# This FINAL version will be used to evaluate the model trained on all three datasets.
###

import os
import json
from tqdm import tqdm

# --- 1. SETUP: DEFINE FILE PATHS ---
print("Initializing script and defining paths...")

# --- Input Paths ---
# The original dev set from FinQA
finqa_dev_path = "data/dev.json"
# The original dev set from TAT-QA
tatqa_dev_path = "data/tatqa_dataset_dev.json"
# The original test set from FiQA (which we use as our dev set)
fiqa_dev_path = "data/fiqa_test.json"

# --- Output Path ---
# The final, combined file for all evaluations.
output_path = "data/final_combined_dev.json"


# --- 2. DEFINE CONVERSION FUNCTIONS ---

def convert_tatqa_to_finqa_format(tatqa_item):
    """
    Converts a single item from the TAT-QA dataset into the simpler FinQA format.
    Returns a list of items, as one TAT-QA entry can contain multiple questions.
    """
    finqa_formatted_items = []
    
    tatqa_table = tatqa_item.get('table', {}).get('table', [])
    finqa_table = {
        "header": tatqa_table[0] if tatqa_table else [],
        "rows": tatqa_table[1:] if len(tatqa_table) > 1 else []
    }

    pre_text = [p['text'] for p in tatqa_item.get('paragraphs', [])]
    post_text = []
    
    for i, qa_pair in enumerate(tatqa_item.get('questions', [])):
        answer_data = qa_pair.get('answer', '')
        if isinstance(answer_data, list):
            answer_str = ", ".join(map(str, answer_data))
        else:
            answer_str = str(answer_data)
            
        new_item = {
            "id": f"tatqa_{tatqa_item.get('uid', '')}_{i}",
            "pre_text": pre_text,
            "post_text": post_text,
            "table": [[finqa_table]],
            "qa": {
                "question": qa_pair.get('question', ''),
                "answer": answer_str
            }
        }
        finqa_formatted_items.append(new_item)
        
    return finqa_formatted_items

def convert_fiqa_to_finqa_format(fiqa_item):
    """
    Converts a single item from the FiQA dataset into the simpler FinQA format.
    """
    instruction = fiqa_item.get("instruction", "")
    question = fiqa_item.get("input", "")
    answer = fiqa_item.get("output", "")

    if not all([instruction, question, answer]):
        return None

    new_item = {
        # Create a unique ID for the FiQA item
        "id": f"fiqa_item_{hash(question)}",
        # FiQA context is in the 'instruction' field. We place it in 'pre_text'.
        "pre_text": [instruction.strip()],
        "post_text": [],
        "table": [], # FiQA data does not contain tables.
        "qa": {
            "question": question,
            "answer": answer
        }
    }
    return new_item

# --- 3. MAIN EXECUTION BLOCK ---

# --- Load FinQA dev data (already in the correct format) ---
print(f"Loading original FinQA dev data from: {finqa_dev_path}")
with open(finqa_dev_path, 'r', encoding='utf-8') as f:
    finqa_dev_data = json.load(f)
print(f"Loaded {len(finqa_dev_data)} entries from FinQA dev set.")

# --- Load and convert TAT-QA dev data ---
print(f"Loading and converting TAT-QA dev data from: {tatqa_dev_path}")
with open(tatqa_dev_path, 'r', encoding='utf-8') as f:
    tatqa_dev_data_raw = json.load(f)

converted_tatqa_data = []
for item in tqdm(tatqa_dev_data_raw, desc="Converting TAT-QA dev set"):
    converted_items = convert_tatqa_to_finqa_format(item)
    converted_tatqa_data.extend(converted_items)
print(f"Converted TAT-QA dev set into {len(converted_tatqa_data)} FinQA-formatted entries.")

# --- Load and convert FiQA dev data ---
print(f"Loading and converting FiQA dev data from: {fiqa_dev_path}")
with open(fiqa_dev_path, 'r', encoding='utf-8') as f:
    fiqa_dev_data_raw = json.load(f)

converted_fiqa_data = []
for item in tqdm(fiqa_dev_data_raw, desc="Converting FiQA dev set"):
    converted_item = convert_fiqa_to_finqa_format(item)
    if converted_item:
        converted_fiqa_data.append(converted_item)
print(f"Converted FiQA dev set into {len(converted_fiqa_data)} FinQA-formatted entries.")


# --- Combine all three datasets ---
combined_dev_data = finqa_dev_data + converted_tatqa_data + converted_fiqa_data
print(f"\nCombined dataset created with a total of {len(combined_dev_data)} entries.")

# --- Save the final combined dev file ---
print(f"Saving final combined dev dataset to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(combined_dev_data, f, indent=2, ensure_ascii=False)

print("\n--- Final combined dev dataset is ready! ---")