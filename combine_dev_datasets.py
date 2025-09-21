###
# combine_dev_datasets.py
# This script combines the validation/development sets from FinQA and TAT-QA
# into a single, unified validation set with a consistent format.
# This is crucial for fairly evaluating models trained on the combined training data.
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

# --- Output Path ---
# The final, combined file that will be used for all future evaluations.
output_path = "data/combined_dev.json"


# --- 2. DEFINE TAT-QA to FinQA CONVERSION FUNCTION ---

def convert_tatqa_to_finqa_format(tatqa_item):
    """
    Converts a single item from the TAT-QA dataset into the simpler FinQA format.
    Returns a list of items, as one TAT-QA entry can contain multiple questions.
    """
    finqa_formatted_items = []
    
    # --- Step A: Reformat the table ---
    # The table structure is slightly different. TAT-QA's table is nested.
    tatqa_table = tatqa_item.get('table', {}).get('table', [])
    finqa_table = {
        "header": tatqa_table[0] if tatqa_table else [],
        "rows": tatqa_table[1:] if len(tatqa_table) > 1 else []
    }

    # --- Step B: Combine paragraphs into pre_text ---
    # We will treat all paragraphs as 'pre_text' for simplicity.
    pre_text = [p['text'] for p in tatqa_item.get('paragraphs', [])]
    post_text = [] # TAT-QA does not have a 'post_text' equivalent.
    
    # --- Step C: Create a separate entry for each question ---
    for i, qa_pair in enumerate(tatqa_item.get('questions', [])):
        # Convert the answer to a string, handling both lists and single values.
        answer_data = qa_pair.get('answer', '')
        if isinstance(answer_data, list):
            answer_str = ", ".join(map(str, answer_data))
        else:
            answer_str = str(answer_data)
            
        new_item = {
            "id": f"tatqa_{tatqa_item.get('uid', '')}_{i}",
            "pre_text": pre_text,
            "post_text": post_text,
            "table": [[finqa_table]], # Match FinQA's nested list structure for the table
            "qa": {
                "question": qa_pair.get('question', ''),
                "answer": answer_str
            }
        }
        finqa_formatted_items.append(new_item)
        
    return finqa_formatted_items

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

# --- Combine the two datasets ---
combined_dev_data = finqa_dev_data + converted_tatqa_data
print(f"\nCombined dataset created with a total of {len(combined_dev_data)} entries.")

# --- Save the final combined dev file ---
print(f"Saving combined dev dataset to: {output_path}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(combined_dev_data, f, indent=2, ensure_ascii=False)

print("\n--- Combined dev dataset is ready! ---")