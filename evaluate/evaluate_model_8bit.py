###
# evaluate_model_8bit.py
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
from tqdm import tqdm

# --- NEW: Function to extract numbers from a string ---
def is_answer_correct(generated_answer, ground_truth):
    # This function compares the generated answer and ground truth by extracting numbers.
    gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
    if not gt_nums:
        return False
    gt_num = float(gt_nums[0])

    gen_nums = re.findall(r'-?\d+\.?\d*', generated_answer.replace(',', ''))
    if not gen_nums:
        return False

    for num_str in gen_nums:
        if abs(float(num_str) - gt_num) < 1e-3:
            return True

    return False

# --- 1. Setup: Load Token and Define Paths (MODIFIED FOR 8-BIT) ---
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded.")

base_model_id = "meta-llama/Llama-2-7b-hf"
# ### CHANGE 1: Point to the 8-bit adapter ###
adapter_path = "./results_lora_8bit_combined/checkpoints/checkpoint-1559"
validation_data_path = "data/final_combined_dev.json"
# ### CHANGE 2: Set a new output file name ###
results_output_path = "evaluation_results_8bit_combined.json"

# --- 2. Load Model with 8-bit Quantization (MODIFIED FOR 8-BIT) ---
# ### CHANGE 3: Use the 8-bit quantization configuration ###
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
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
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def create_inference_prompt(sample):
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

print(f"Loading validation data from: {validation_data_path}")
with open(validation_data_path, 'r', encoding='utf-8') as f:
    validation_data = json.load(f)

# --- NEW: Batch Inference Implementation ---
results = []
correct_predictions = 0
batch_size = 8  # GPU 메모리에 따라 4, 8, 16 등으로 조절

print(f"\nStarting evaluation on {len(validation_data)} samples with batch size {batch_size}...")

# tqdm을 사용하여 전체 데이터셋에 대한 진행률 표시
for i in tqdm(range(0, len(validation_data), batch_size), desc="Evaluating"):
    # 1. 처리할 배치 데이터 슬라이싱
    batch_samples = validation_data[i:i + batch_size]
    
    # 2. 배치 전체에 대한 프롬프트 생성
    batch_prompts = [create_inference_prompt(sample) for sample in batch_samples]

    # 3. 배치 토크나이징 (padding 추가)
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    ).to("cuda")

    # 4. 배치 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # 5. 배치 디코딩
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # 6. 배치 결과 처리
    for j, decoded_output in enumerate(decoded_outputs):
        sample = batch_samples[j]
        prompt = batch_prompts[j]
        
        generated_answer_text = decoded_output[len(prompt):].strip()
        ground_truth_text = str(sample.get('qa', {}).get('answer', ''))
        is_correct = is_answer_correct(generated_answer_text, ground_truth_text)

        if is_correct:
            correct_predictions += 1

        results.append({
            "id": sample.get("id"),
            "question": sample.get('qa', {}).get('question'),
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
