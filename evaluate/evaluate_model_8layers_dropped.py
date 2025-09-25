###
# evaluate_model_fullft.py
import os
import json
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from dotenv import load_dotenv
from tqdm import tqdm

# --- Function to extract numbers from a string for numeric matching ---
def is_answer_correct(generated_answer, ground_truth):
    # Extracts the first number from the ground truth
    gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
    if not gt_nums:
        return False
    gt_num = float(gt_nums[0])

    # Checks if any number in the generated answer matches the ground truth number
    gen_nums = re.findall(r'-?\d+\.?\d*', generated_answer.replace(',', ''))
    if not gen_nums:
        return False

    for num_str in gen_nums:
        if abs(float(num_str) - gt_num) < 1e-3: # Using a small tolerance for float comparison
            return True

    return False

# --- 1. Setup: Load Token and Define Paths ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# ### CHANGE 1: Point to the fully fine-tuned model directory ###
fully_tuned_model_path = "./results_sapling/pruned_8_layers/checkpoints/checkpoint-390"
validation_data_path = "data/dev.json"
# ### CHANGE 2: Set a new output file name for the results ###
results_output_path = "evaluation_results_sapling_8_layers.json"

# --- 2. Load Fully Fine-Tuned Model in BF16 Precision ---
print(f"Loading fully fine-tuned model from: {fully_tuned_model_path}")

# ### CHANGE 3: Load the fine-tuned model directly. No base model or adapter needed. ###
model = AutoModelForCausalLM.from_pretrained(
    fully_tuned_model_path,
    torch_dtype=torch.bfloat16, # Use bfloat16 to match training precision
    device_map="auto",
    token=hf_token
)
# The tokenizer is saved with the model, so we load it from the same directory
tokenizer = AutoTokenizer.from_pretrained(fully_tuned_model_path, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.eval()
print("Model and tokenizer loaded successfully.")


# --- (The rest of the script is identical as it's model-agnostic) ---
def create_inference_prompt(sample):
    # This robust prompt creation handles potential inconsistencies in the FinQA dataset structure
    table_string = ""
    if sample.get('table') and isinstance(sample.get('table'), list) and len(sample.get('table')) > 0:
        table_data = sample['table'][0]
        if isinstance(table_data, list) and table_data:
            table_data = table_data[0]
        if isinstance(table_data, dict) and table_data.get('header') and table_data.get('rows'):
            header = " | ".join(table_data['header'])
            rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
            table_string = f"TABLE:\n{header}\n{rows}"
    pre_text = "\n".join(sample.get('pre_text', []))
    post_text = "\n".join(sample.get('post_text', []))
    qa_data = sample.get('qa', [{}])[0] if isinstance(sample.get('qa'), list) and sample.get('qa') else sample.get('qa', {})
    question = qa_data.get('question', 'QUESTION_NOT_FOUND')
    prompt = f"""### INSTRUCTION:
Answer the question based on the context below.
### CONTEXT:
{pre_text}
{table_string}
{post_text}
### QUESTION:
{question}
### ANSWER:"""
    return prompt

print(f"Loading validation data from: {validation_data_path}")
with open(validation_data_path, 'r', encoding='utf-8') as f:
    validation_data = json.load(f)

results = []
correct_predictions = 0

print(f"\nStarting evaluation on {len(validation_data)} samples...")
for sample in tqdm(validation_data, desc="Evaluating"):
    prompt = create_inference_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer_text = decoded_output[len(prompt):].strip()
    ground_truth_text = str(sample.get('qa', {}).get('answer', '')) if isinstance(sample.get('qa'), dict) else str(sample.get('qa', [{}])[0].get('answer', ''))

    is_correct = is_answer_correct(generated_answer_text, ground_truth_text)

    if is_correct:
        correct_predictions += 1

    results.append({
        "id": sample.get("id"),
        "question": sample.get('qa', {}).get('question', 'N/A'),
        "ground_truth": ground_truth_text,
        "generated_answer": generated_answer_text,
        "is_correct": is_correct
    })

accuracy = (correct_predictions / len(validation_data)) * 100 if validation_data else 0

print("\n--- Evaluation Complete ---")
print(f"Total Samples: {len(validation_data)}")
print(f"Correct Predictions (Numeric Match): {correct_predictions}")
print(f"Numeric Match Accuracy: {accuracy:.2f}%")
print("---------------------------")

print(f"Saving detailed results to {results_output_path}...")
with open(results_output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Results saved.")