###
# preprocess_data_paranoid_safe.py
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
from tqdm import tqdm

def create_prompt(data_item):
    # This paranoid function handles multiple possible data structures.
    
    sample = {}
    # Case 1: The data item is a dictionary (the expected format).
    if isinstance(data_item, dict):
        sample = data_item
    # Case 2: The data item is a list containing a single dictionary.
    elif isinstance(data_item, list) and len(data_item) > 0 and isinstance(data_item[0], dict):
        sample = data_item[0]
    else:
        # If the format is completely unexpected, return an error message.
        return {"text": f"### ERROR: UNPROCESSABLE DATA ITEM ###\nType: {type(data_item)}\nContent: {str(data_item)[:200]}"}

    # From here, we are sure 'sample' is a dictionary.
    # We will use .get() with default values everywhere to prevent errors.
    
    table_string = ""
    table_list = sample.get('table', []) # Default to an empty list
    if isinstance(table_list, list) and len(table_list) > 0:
        table_data = table_list[0]
        if isinstance(table_data, dict):
            header = " | ".join(table_data.get('header', []))
            rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
            table_string = f"TABLE:\n{header}\n{rows}"

    pre_text = "\n".join(sample.get('pre_text', []))
    post_text = "\n".join(sample.get('post_text', []))
    
    qa_data = sample.get('qa', {})
    question = qa_data.get('question', 'QUESTION_NOT_FOUND')
    answer = qa_data.get('answer', 'ANSWER_NOT_FOUND')

    prompt = f"""### INSTRUCTION:
Answer the question based on the context below.

### CONTEXT:
{pre_text}
{table_string}
{post_text}

### QUESTION:
{question}

### ANSWER:
{answer}"""
    
    return {"text": prompt}

# --- Main Execution Block ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("Hugging Face Token not found.")
else:
    print("Complete. Hugging Face Token Ready")

model_id = "meta-llama/Llama-2-7b-hf"
local_data_path = "data/train.json"
output_path = "data/processed_train.json"

try:
    print(f"'{model_id}' tokenizer is loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    print("Tokenizer loading complete!")

    print(f"Loading the local json file from: '{local_data_path}'...")
    with open(local_data_path, 'r', encoding='utf-8') as f:
        finqa_data = json.load(f)
    print("JSON loading complete.")
    
    print("\nFormatting dataset with prompt template...")
    processed_data = []
    for item in tqdm(finqa_data, desc="Processing samples"):
        processed_data.append(create_prompt(item))
    
    print("Formatting complete!")

    print(f"Saving processed data to '{output_path}'...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print("Saving complete!")

    final_dataset = Dataset.from_list(processed_data)
    print("Dataset object creation complete!")

    print("\n--- Preprocessed Sample Text ---")
    print(final_dataset[0]['text'])
    print("--------------------------------")
    print(f"\nSuccessfully processed {len(final_dataset)} samples.")
    print("Phase 1-2: Data Preprocessing is complete!")

except Exception as e:
    print(f"An error has occurred: {e}")