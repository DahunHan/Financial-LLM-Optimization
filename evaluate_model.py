###
# evaluate_model.py

# --- 1. Import Necessary Libraries ---
import os
import json
import torch
import re
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from dotenv import load_dotenv
# # Import tqdm for a nice progress bar during the loop.
from tqdm import tqdm

# NEW : Robust evaluation function
def is_answer_correct(generated_answer, ground_truth):
    ## Compares the generated answer and ground truth by extracting all numbers 
    ## and checking if the most likely number from the answer matches the ground truth number.

    # Find all numbers (including decimals and negatives) in the ground truth.
    # We assue the ground truth is a clean number, but we check just in case.
    gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
    if not gt_nums:
        return False
    gt_num = float(gt_nums[0])

    # Find all numbers in the model's generated answer
    gen_nums = re.findall(r'-?\d+\.?\d*', generated_answer.replace(',', ''))
    if not gen_nums:
        return False
    # Check if the exact ground truth number exists in the generated numbers
    for num_str in gen_nums:
        if abs(float(num_str) - gt_num) < 1e-3:
            return True
    return False

# --- 2. Setup: Load Token and Define Paths ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

# # Define the base model and the path to our trained adapter.
# # You can switch this to "./results_8bit/final_model" to evaluate the 8-bit model.
base_model_id = "meta-llama/Llama-2-7b-hf"
adapter_path = "./results_4bit/final_model"
validation_data_path = "data/dev.json"
results_output_path = "evaluation_results_4bit.json"

# --- 3. Load the Quantized Base Model and Adapter ---
# # It's crucial to use the same quantization config as in training.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading base model: {base_model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading and merging LoRA adapter from: {adapter_path}")
# # Merge the base model with the trained adapter to create our final fine-tuned model.
model = PeftModel.from_pretrained(base_model, adapter_path)
# # Set the model to evaluation mode.
model.eval()

# --- 4. Define the Inference Prompt Function ---
def create_inference_prompt(sample):
    # # This function formats a raw data sample into a text prompt for the model.
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

# --- 5. Load Validation Data ---
print(f"Loading validation data from: {validation_data_path}")
with open(validation_data_path, 'r', encoding='utf-8') as f:
    validation_data = json.load(f)

# --- 6. Run Evaluation Loop ---
results = []
correct_predictions = 0

print(f"\nStarting evaluation on {len(validation_data)} samples...")
# # Use tqdm to create a progress bar for the loop.
for sample in tqdm(validation_data, desc="Evaluating"):
    # # Prepare the prompt for the current sample.
    prompt = create_inference_prompt(sample)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # # Generate an answer from the model.
    with torch.no_grad(): # # Disable gradient calculation for faster inference.
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
    
    # # Decode the output and extract the answer part.
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer_text = decoded_output[len(prompt):].strip()
    # # Get the actual correct answer from the dataset.
    ground_truth_answer = str(sample.get('qa', {}).get('answer', ''))

    is_correct = is_answer_correct(generated_answer_text, ground_truth_answer)

    if is_correct:
        correct_predictions += 1

    # # Store the results for this sample.
    results.append({
        "id": sample.get("id"),
        "question": sample.get('qa', {}).get('question'),
        "ground_truth": ground_truth_answer,
        "generated_answer": generated_answer_text,
        "is_correct": is_correct
    })

# --- 7. Calculate and Save Final Results ---
# # Calculate the final accuracy.
accuracy = (correct_predictions / len(validation_data)) * 100

print("\n--- Evaluation Complete ---")
print(f"Total Samples: {len(validation_data)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Exact Match Accuracy: {accuracy:.2f}%")
print("---------------------------")

# # Save the detailed results to a JSON file.
print(f"Saving detailed results to {results_output_path}...")
with open(results_output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Results saved.")