import os
import json
from datasets import load_dataset, Features, Value, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

#.env 파일에서 환경 변수(HF_TOKEN)를 불러옴
load_dotenv()

#Hugging Face 토큰을 환경 변수에서 가져옴
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("Can't find Hugging Face Token, please check your .env file")
else:
    print("Complete. Hugging Face Token Ready")

#Model & Dataset That I plan to use
model_id = "meta-llama/Llama-2-7b-hf"
local_data_path = "data/train.json"

try:
    #Call Model Tokenizer
    print(f"'{model_id}' tokenizer is loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    print("Tokenizer loading complete!")

    print(f"Loading the local dataset from: '{local_data_path}'")
    
    #Open file with python's basic json library.
    with open(local_data_path, 'r', encoding = 'utf-8') as f:
        finqa_data = json.load(f)
    print("Preprocessing data: converting all qa fields to string...")
    for item in finqa_data:
        # 'qa' 딕셔너리의 모든 값을 문자열로 변환합니다.
        for key, value in item['qa'].items():
            item['qa'][key] = str(value)
    print("Preprocessing complete!")
    #Create Dataset features based on recalled data list.
    dataset= Dataset.from_list(finqa_data)

    #Checking first sample in streaming data
    sample = dataset[0]
    print("Data set load complete")
    print("\n--- FinQA dataset sample")
    print(sample)
    print("-------------------")

    print("\nModel & FinQA Dataset Load test Complete")

except Exception as e:
    print(f"Error has occured: {e}")