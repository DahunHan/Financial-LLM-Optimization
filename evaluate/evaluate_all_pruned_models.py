###
# evaluate_all_pruned_models.py
# This script automatically evaluates a list of model checkpoints from the successive pruning experiment.
###

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
    gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
    if not gt_nums: return False
    gt_num = float(gt_nums[0])

    gen_nums = re.findall(r'-?\d+\.?\d*', generated_answer.replace(',', ''))
    if not gen_nums: return False

    for num_str in gen_nums:
        if abs(float(num_str) - gt_num) < 1e-3:
            return True
    return False

# --- Setup: Load Token ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# --- ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---
# --- 1. DEFINE ALL MODEL CHECKPOINTS TO EVALUATE ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---
model_paths_to_evaluate = [
    # Add the baseline model (0 layers dropped) for a complete comparison
    "./results_successive/run_0_drop_0/checkpoint-1559", 
    "./results_successive/run_1_drop_1/checkpoint-1559",
    "./results_successive/run_2_drop_2/checkpoint-1559",
    "./results_successive/run_3_drop_3/checkpoint-1559",
    "./results_successive/run_4_drop_4/checkpoint-1559",
    "./results_successive/run_5_drop_5/checkpoint-1559",
    "./results_successive/run_6_drop_6/checkpoint-1559",
]

# --- Load validation data ONCE for efficiency ---
validation_data_path = "data/final_combined_dev.json"
print(f"Loading validation data from: {validation_data_path}")
with open(validation_data_path, 'r', encoding='utf-8') as f:
    validation_data = json.load(f)
print("Validation data loaded successfully.")

def create_inference_prompt(sample):
    table_string = ""
    if sample.get('table') and isinstance(sample.get('table'), list) and len(sample.get('table')) > 0:
        table_data = sample['table'][0]
        if isinstance(table_data, list) and table_data: table_data = table_data[0]
        if isinstance(table_data, dict) and table_data.get('header') and table_data.get('rows'):
            header = " | ".join(table_data['header'])
            rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
            table_string = f"TABLE:\n{header}\n{rows}"
    pre_text = "\n".join(sample.get('pre_text', []))
    post_text = "\n".join(sample.get('post_text', []))
    question = sample.get('qa', {}).get('question', 'QUESTION_NOT_FOUND')
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

# --- ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---
# --- 2. MAIN LOOP TO EVALUATE EACH MODEL ---
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---
for model_path in model_paths_to_evaluate:
    print("\n" + "="*80)
    print(f"STARTING EVALUATION FOR MODEL: {model_path}")
    print("="*80)

    # --- Dynamically generate the output path for each model's results ---
    run_name = model_path.split('/')[-2]  # Extracts "run_X_drop_X"
    results_output_path = f"evaluation_results_{run_name}.json"
    print(f"Results will be saved to: {results_output_path}")

    # --- Load the specific fine-tuned model for this iteration ---
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    print("Model and tokenizer loaded successfully.")

    # --- Run the evaluation loop for the current model ---
    results = []
    correct_predictions = 0
    batch_size = 8 # Using batching for speed

    for i in tqdm(range(0, len(validation_data), batch_size), desc=f"Evaluating {run_name}"):
        batch_samples = validation_data[i:i + batch_size]
        batch_prompts = [create_inference_prompt(sample) for sample in batch_samples]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, decoded_output in enumerate(decoded_outputs):
            sample = batch_samples[j]
            prompt = batch_prompts[j]
            generated_answer_text = decoded_output[len(prompt):].strip()
            ground_truth_text = str(sample.get('qa', {}).get('answer', ''))

            if is_answer_correct(generated_answer_text, ground_truth_text):
                correct_predictions += 1

            results.append({
                "id": sample.get("id"),
                "question": sample.get('qa', {}).get('question', 'N/A'),
                "ground_truth": ground_truth_text,
                "generated_answer": generated_answer_text,
                "is_correct": is_answer_correct(generated_answer_text, ground_truth_text)
            })

    accuracy = (correct_predictions / len(validation_data)) * 100 if validation_data else 0

    print("\n" + f"--- Evaluation Complete for {run_name} ---")
    print(f"Total Samples: {len(validation_data)}")
    print(f"Correct Predictions (Numeric Match): {correct_predictions}")
    print(f"Numeric Match Accuracy: {accuracy:.2f}%")
    print("--------------------------------" + "-"*len(run_name))

    print(f"Saving detailed results to {results_output_path}...")
    with open(results_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Results saved.")

print("\n\nAll model evaluations are complete!")