Financial LLM Optimization via Structural Compression
A research project on enhancing the efficiency and accuracy of Large Language Models (LLMs) for the financial domain. This project systematically evaluates structural compression as a superior alternative to conventional methods for optimizing models on general-purpose hardware.

<br>

1. Project Goal & Core Hypothesis
The goal of this research is to identify and validate the most effective optimization strategies for LLMs in the specialized financial domain. We aim to maximize practical inference efficiency (i.e., speed and memory usage) on consumer-grade and cloud hardware, while maintaining the high level of accuracy required for complex financial tasks.

Our core hypothesis is as follows:

The financial domain demands exceptional accuracy from LLMs. Conventional compression techniques like Quantization can be detrimental, as they uniformly degrade model parameters, potentially compromising the critical, complex reasoning capabilities essential in finance. In contrast, we hypothesize that Structural Compression methods, such as Layer Dropping, will achieve a superior trade-off. By selectively removing entire layers (knowledge modules) deemed less relevant to the financial domain, this approach can secure significant, hardware-agnostic efficiency gains while better preserving the model's core reasoning accuracy.

<br>

2. Methodology
To validate our hypothesis, this research will implement and compare the following methodologies using Llama-2-7B on the FinQA dataset.

Proposed Method: SAPLING (Structural Compression)

Description: A framework that performs domain-specific adaptation and compression simultaneously. Its core technique is Successive Layer Dropping, which progressively reduces the model's depth.

Role: The novel, efficient optimization technique whose performance-efficiency trade-off is the subject of this study.

Baseline 1: Full Fine-Tuning (Performance Ceiling)

Description: A standard fine-tuning approach where all model parameters are updated.

Role: To establish the theoretical maximum performance achievable on the FinQA dataset, serving as the gold-standard accuracy benchmark against which all other methods are measured.

Baseline 2: LoRA & QLoRA (Efficiency Baselines)

Description: Parameter-Efficient Fine-Tuning (PEFT) methods that update only a small subset of parameters (LoRA) or operate on a quantized base model (QLoRA).

Role: To represent the current industry standard for efficient fine-tuning. SAPLING's performance will be compared against these baselines to demonstrate its relative advantages.

<br>

3. Experimental Plan
We will conduct a systematic, multi-phase experimental process to test our hypothesis.

Phase 1: Establish Performance Baselines

Objective: To fine-tune the Llama-2-7B model using various standard methods to create a comprehensive set of performance and efficiency benchmarks.

Experiments:

Full Fine-Tuning (16-bit, on A100 GPU): Establish the performance ceiling.

LoRA Fine-Tuning (16-bit, on A100 GPU): Establish a high-quality efficiency baseline.

LoRA Fine-Tuning (8-bit, on RTX 3080/A100): Measure performance under moderate memory constraints.

QLoRA Fine-Tuning (4-bit, on RTX 3080): Measure performance under severe memory constraints.

Key Metrics: QA Accuracy (%), Max VRAM Usage (GB), Total Training Time, eval_loss curve.

Phase 2: Evaluate Proposed Method (SAPLING)

Objective: To implement and apply the SAPLING framework and rigorously compare its performance against the established baselines.

Key Metrics: A Pareto frontier graph of Model Size (%) vs. Accuracy (%), and the final inference throughput (tokens/sec) to visualize the trade-offs.

<br>

4. Results
This section will be updated as the experiments are completed.

Method

Precision

Accuracy (%)

Model Size (GB)

Training Hardware

Full Fine-Tuning

16-bit (BF16)

TBD

~28 GB

A100

LoRA

16-bit (FP16)

1.47%

~28 GB + Adapter

A100

LoRA

8-bit

TBD

~14 GB + Adapter

A100

QLoRA

4-bit

1.02%

~7 GB + Adapter

RTX 3080

SAPLING (50% Drop)

16-bit

TBD

TBD

A100

<br>

5. Setup & How to Run
This research can be reproduced using the following steps.

1. Environment Setup

# Clone the repository
git clone https://github.com/DahunHan/Financial-LLM-Optimization.git
cd Financial-LLM-Optimization

# Create and activate the Python virtual environment
# Note: This project requires Python 3.10 for library compatibility.
python3.10 -m venv .venv
source .venv/bin/activate # on Linux/macOS
# .\.venv\Scripts\activate # on Windows

# Install dependencies
pip install -r requirements.txt

2. API Keys & Wandb
Create a .env file in the root directory and add your Hugging Face Hub token. Log in to Weights & Biases for experiment tracking.

# .env file content
HUGGING_FACE_HUB_TOKEN="hf_..."

# Terminal command
wandb login

3. Running Experiments

# 1. Run the data preprocessing script first
python preprocess_data.py

# 2. Run a training script (choose one)
python train_full_ft.py        # For Full Fine-Tuning (requires A100+)
python train_lora_16bit.py     # For 16-bit LoRA (requires A100+)
python train_lora_8bit.py      # For 8-bit LoRA
python train_qlora_4bit.py     # For 4-bit QLoRA

<br>

6. Hardware Analysis & Environment Notes
This research is conducted across two hardware setups, which directly informs the project's methodology.

Local Machine: NVIDIA RTX 3080 (10GB VRAM)

Cloud Instance: Lambda Labs A100 (40GB VRAM)

VRAM Analysis for Llama-2-7B (Experimental Results):

Full Fine-Tuning (16-bit): Impossible on RTX 3080. Requires ~28GB VRAM just to load the model and optimizer states, far exceeding the available 10GB. This experiment is exclusively run on the A100 GPU.

8-bit LoRA Fine-Tuning: Possible but slow on RTX 3080. This is only achievable by enabling Gradient Checkpointing, which trades computation time for memory.

4-bit QLoRA Fine-Tuning: Most efficient method for local fine-tuning. Quantizing the base model to 4-bit reduces its memory footprint to ~3.5GB, providing sufficient headroom for a stable and fast training process on the RTX 3080.

These experimentally verified trade-offs are central to the project, highlighting the need for advanced compression techniques like SAPLING to enable high-performance model training on more accessible hardware.