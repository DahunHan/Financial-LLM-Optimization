###
# preprocess_tatqa.py
# This script preprocesses the raw TAT-QA dataset into the simple
# {"text": "..."} format, making it compatible with our existing training scripts.
###

import os
import json
from tqdm import tqdm

# --- 1. SETUP: DEFINE FILE PATHS ---
print("Initializing script and defining paths...")
# Input path for the raw TAT-QA data.
input_data_path = "data/tatqa_dataset_train.json"

# The output path for the processed data.
output_data_path = "data/processed_tatqa_train.json"

# --- 2. DEFINE DATA PROCESSING FUNCTIONS ---

def format_table(table_data):
    """Formats the table dictionary into a readable string."""
    try:
        header = " | ".join(table_data.get('header', []))
        rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
        return f"TABLE:\n{header}\n{rows}"
    except Exception:
        return "TABLE:\n[Error formatting table]"

def create_prompt_for_tatqa(item):
    """
    Takes a single item from the TAT-QA dataset and yields a list of
    fully formatted text prompts, one for each question in the item.
    """
    # Format the context (paragraphs and table)
    paragraphs = "\n".join([p['text'] for p in item.get('paragraphs', [])])
    table_str = format_table(item.get('table', {}))
    context = f"{paragraphs}\n\n{table_str}".strip()

    prompts = []
    # Iterate through all questions associated with this context.
    for qa_pair in item.get('questions', []):
        question = qa_pair.get('question', 'N/A')
        
        # TAT-QA answers can be a list or a single value. We handle both.
        answer_data = qa_pair.get('answer', 'N/A')
        if isinstance(answer_data, list):
            answer = ", ".join(map(str, answer_data))
        else:
            answer = str(answer_data)

        # Build the final prompt text in our standard format.
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

print(f"Loading raw TAT-QA data from: {input_data_path}")
with open(input_data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

processed_data = []
print(f"Processing {len(raw_data)} entries from TAT-QA...")

for item in tqdm(raw_data, desc="Preprocessing TAT-QA"):
    # The create_prompt function returns a list of prompts for each item.
    prompts_for_item = create_prompt_for_tatqa(item)
    processed_data.extend(prompts_for_item)

print(f"Generated a total of {len(processed_data)} processed data points.")

# --- 4. SAVE THE PROCESSED DATA ---
print(f"Saving processed data to: {output_data_path}")
with open(output_data_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

print("\nTAT-QA preprocessing complete!")
