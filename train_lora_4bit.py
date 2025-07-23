#train_lora_8bit.
#May face an "Out of Memory" Error.
#If so, blame it on my poor GPU :(
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

## 1. Load hugging face token
load_dotenv()
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
print("Hugging face token lock and loaded")

## 2. Model and Data Paths
model_id = "meta-llama/Llama-2-7b-hf"
processed_data_path = "data/processed_train.json"

## 3. Load Model with 8-bit Quantization 
### You got this MY 3080
print(f"Loading base model : {model_id} with 4-bit QLoRA precision")

### 4-bit config.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16,
bnb_4bit_use_double_quant=True,
)
### Download and load Llama 2 model from hub
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map ="auto",
    token=hf_token
)
### Disable caching of past key/value states during text generation
### Not required for training moreover we can save space
model.config.use_cache = False
### Compatibility Setting. Setting this to 1 tells the model we are NOT using tensor parallelism
model.config.pretraining_tp=1
### Preapre the model for k-bit training
model = prepare_model_for_kbit_training(model)

## 4. Load Tokenizer
### Tokenizer is responsible for converting raw text into numbers(token) 
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token = hf_token)
### Llama 2 doesn't have a default padding token.
### Thus we must set it to be the same as the end-of-sentence token.
tokenizer.pad_token=tokenizer.eos_token
### For casual language models, padding should be on the right to avoid confusion.
tokenizer.padding_side = "right"

## 5. Load Dataset
print(f"Loading processed data set from : {processed_data_path}")
dataset = load_dataset("json", data_files=processed_data_path, split="train")
### Split="Train" tells the library to treat the entire file as the training set.
print(f"Succesfully loaded {len(dataset)} samples.")

def tokenize_function(examples):
    # ...and uses the tokenizer to convert the 'text' field into numerical tokens.
    # truncation=True: cuts off text that is longer than max_length.
    # padding="max_length": adds padding tokens to text that is shorter than max_length.
    # max_length=512: sets the uniform length for all sequences.
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# The .map() function applies the tokenize_function to every sample 
# batched = True processes multiple samples at once. Makes it faster simply.
tokenized_dataset = dataset.map(tokenize_function, batched = True)
# Remove the original 'text' column as it is no longer needed
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
print(f"Succesfully loaded and tokenized {len(tokenized_dataset)} samples")

## 6. PEFT Configuration (LoRA)
### Configure the LoRA adapters that will be added to the model.
lora_config = LoraConfig(
    #scaling factor for LoRA weights. 
    lora_alpha = 16,
    #dropout is a regularization technique to prevent overfitting in the LoRA layers
    lora_dropout=0.1,
    # "r" is the rank of the LoRA matrices. 
    r = 64,
    # Won't be training the bias parameters, only weights.
    bias = "none",
    # Causal language modeling task (NOT CASUAL)
    task_type= "CAUSAL_LM",
)
### Wrap our base model with the LoRA adapters
### Make only weights trainable.
model = get_peft_model(model, lora_config)

## 7. Training Arguments
### Let us hold all the hyperparametrs and settings for the training process.
training_args = TrainingArguments(
    # Directory where the training outputs will be saved.
    output_dir = "./results_4bit",
    # Total number of times the trainer (I) will iterate through the entire dataset
    num_train_epochs= 1,
    # Number of training samples to process in a single batch on one device
    # Set this to 1 to be conservative with my VRAM usage OTL
    per_device_train_batch_size=1,
    # A technique to simulate a larger batch size. Gradients are accumulated for this
    # many steps before a model update is performed
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # Learning rate for the optimizer
    learning_rate=2e-4,
    # Use 16-bit floating bit point precision (mixed precision training)
    # This speeds up the training and reduces memory usage without sig. loss of accuracy.(Hopefully)
    fp16 = True,
    # How often to log training progress to console
    logging_steps=10,
    # Initial test, we will only run a maximum of 50 training steps 
    # to ensure the setup works.
    max_steps=50,
    # Save a check point of the model every 25 steps
    save_steps=25,
    remove_unused_columns=False,
)

## 8. Initialize Trainer with Data Collator
### The Data collator takes a list of samples from the dataset and groups them into a batch
### it also handles padding dynacmically, which can be more efficient
### mlm=False means we are doing Causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

### The trainer class is a great tool from Hugging Face
### It handles the entire training loop, optimization, logging and saving 
trainer = Trainer(
    model = model,
    # Training argument that we just defined in step 7
    args = training_args,
    # dataset defined in step 5
    train_dataset=tokenized_dataset,
    # Tokenizer required to prepare the data defined in step 4
    tokenizer = tokenizer,
    # For a text completion task like this, a custom 'data collator' is often needed
    data_collator = data_collator,
)

## 9. Start Training
print("\nStarting model training")

trainer.train()
print("Training Complete. I got a bit smarter")

## 10. Save
final_model_path = "./results_4bit/final_model"
trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")