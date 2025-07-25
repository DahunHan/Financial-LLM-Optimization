### inference py

# 1. Import Libraries
import os
import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# PeftModel class -> Specifically designed to handel LoRA adapters.
from peft import PeftModel
from dotenv import load_dotenv

# 2. Setup : Load Token and Define Paths
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging Face Token Loaded")

# Define the identifier for the model
base_model_id = "meta-llama/Llama-2-7b-hf"
# Define the path to the folder 
# Lets start with 4-bit
adapter_path = "./results_4bit/final_model"
# Define path to validation data file.
validation_data_path = "data/dev.json"

# 3. Load the Quantized Base Model
# The base model with the EXACT SAME quantization settings MUST be loaded
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print (f"Loading base model : {base_model_id}")
# Load original Llama 2 model applying the 4bit quantization configuration on the fly.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config = bnb_config,
    device_map="auto", #Automatically mad the model to the GPU
    token=hf_token,
)

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code = True,
    token=hf_token,
)
# Set the padding token to be the same as end-of-sentence token + right side
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. Merge the LoRA Adapter with the Base Model
print(f"Loading and merging LoRA adapter from {adapter_path}")
# Key Step. PeftModel. takes the base model and merges the trained LoRA adapter weights on top of it
model = PeftModel.from_pretrained(base_model, adapter_path)

# 5. Prepare a test sample and prompt
print(f"Loading test data from : {validation_data_path}")
# open and load the validation json file.
with open(validation_data_path, 'r', encoding = 'utf-8') as f:
    validation_data = json.load(f)


# Use first sample from validation data for testing
test_sample = validation_data[0]

# Lets create inference prompt. 
# Its nearly identical to our preprocessing script.
# Difference is tha the ANSWER section is left empty. (Obviously)
def create_inference_prompt(sample):
    table_string = ""
    table_list = sample.get('table',[])
    if table_list:
        table_data_inner = table_list[0]
        # Check if inner element is ALSO a list
        if isinstance(table_data_inner, list) and table_data_inner:
            table_data = table_data_inner[0]
        else:
            table_data = table_data_inner

        # Now we are sure that the 'table_data' is the dictionary we want
        if isinstance(table_data, dict) and table_data.get('header') and table_data.get('rows'):
            header = " | " .join(table_data['header'])
            rows = "\n".join([" | ".join(map(str, row)) for row in table_data.get('rows', [])])
            table_string = f"TABLE:\n{header}\n{rows}"

    pre_text = "\n".join(sample.get('pre_text', []))
    post_text = "\n".join(sample.get('post_text', []))
    # Access the question from the 'qa' field
    question = sample.get('qa', {}).get('question', 'QUESTION_NOT_FOUND')
    qa_data = sample.get('qa', {})
    if isinstance(qa_data, dict):
        question = qa_data.get('question', 'QUESTION_NOT_FOUND')
    elif isinstance(qa_data, list) and qa_data:
        question = qa_data[0].get('question', 'QUESTION_NOT_FOUND')

# Create the exact same prompt structure but leave the answer blank for the modewl to fill in

    prompt = f"""### INSTRUCTION:
Answer the question based on the context below

### CONTEXT :
{pre_text}
{table_string}
{post_text}

### Question:
{question}

### Answer:"""
    return prompt

# Create the final prompt string for our test sample.
inference_prompt = create_inference_prompt(test_sample)
print("\n--- Generated Inference Prompt ---")
print(inference_prompt)
print("------------------------------------")

# 6. Run Model Inference
# Set model to eval mode. 
# This will disable layers like dropout that are only used during training
model.eval()
# Take prompt string and convert into numerical tokens using the tokenizer
# move it to the GPU ('cuda') for processing.
inputs = tokenizer(inference_prompt, return_tensors = "pt").to("cuda")
print("\nModel is generating an answer")

# Core of inference
# Model.generate() function takes the input tokens and predicts the next tokens in the sequence
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=50)
# max new tokens = 50 limits the answer to a maximum of 50 new tokens to prevent long, rambling responses.

# Outputs are a tensor of token IDs. We use the tokenizer's decode function to convert these numbers back into human-readable text
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# skip_special_tokens=True removes tokens like <eos> from the final output.

print("\n--- Model's Full Generated Output --- ")
print(decoded_output)
print("------------------------------------")

# Decoded output contains our original prompt plus the new answer
# To get JUST the answer, we will slice the string, removing the part that matches our input prompt
answer_only = decoded_output[len(inference_prompt):].strip()
# .strip() removes any  leading/trailing whitespace.
print("\n-- Clean Answer Generated by Model ---")
print(answer_only)
print("--------------------------------------------")

# Finally print the correct answer from our dataset for comparison
print("\n--- Actual Correct Asnwer---")
print(test_sample.get('qa', {}).get('answer', 'ANSWER_NOT_FOUND'))
print("--------------------------------------------")