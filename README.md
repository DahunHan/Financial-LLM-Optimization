# Financial LLM Optimization via Structural Compression

A research project on enhancing the efficiency and accuracy of Large Language Models (LLMs) for the financial domain. This project systematically evaluates structural compression as a superior alternative to conventional methods for optimizing models on general-purpose hardware.

<br>

## 1. Project Goal & Core Hypothesis

The goal of this research is to identify and validate the most effective optimization strategies for LLMs in the specialized financial domain. We aim to maximize **practical inference efficiency** (i.e., speed and memory usage) on **consumer-grade and cloud hardware**, while maintaining the **high level of accuracy** required for complex financial tasks.

Our core hypothesis is as follows:

> The financial domain demands exceptional accuracy from LLMs. Conventional compression techniques like **Quantization** can be detrimental, as they uniformly degrade model parameters, potentially compromising the critical, complex reasoning capabilities essential in finance. In contrast, we hypothesize that **Structural Compression** methods, such as **Layer Dropping**, will achieve a superior trade-off. By selectively removing entire layers (knowledge modules) deemed less relevant to the financial domain, this approach can secure significant, hardware-agnostic efficiency gains while better preserving the model's core reasoning accuracy.

<br>

## 2. Methodology

To validate our hypothesis, this research will implement and compare the following methodologies using `Llama-2-7B` on the `FinQA` dataset.

* **Proposed Method: SAPLING (Structural Compression)**
    * **Description:** A framework that performs domain-specific adaptation and compression simultaneously using Layer Dropping. We will investigate two distinct pruning strategies.
    * **Role:** The novel, efficient optimization technique whose performance-efficiency trade-off is the subject of this study.

* **Baseline 1: Full Fine-Tuning (Performance Ceiling)**
    * **Description:** A standard fine-tuning approach where all model parameters are updated.
    * **Role:** To establish the **theoretical maximum performance** achievable on the FinQA dataset, serving as the gold-standard accuracy benchmark.

* **Baseline 2: LoRA & QLoRA (Efficiency Baselines)**
    * **Description:** Parameter-Efficient Fine-Tuning (PEFT) methods that update only a small subset of parameters (LoRA) or operate on a quantized base model (QLoRA).
    * **Role:** To represent the **current industry standard for efficient fine-tuning**.

<br>

## 3. Experimental Plan

* **Phase 1: Establish Performance Baselines**
    * **Objective:** To fine-tune the Llama-2-7B model using various standard methods to create a comprehensive set of performance and efficiency benchmarks.
    * **Experiments:**
        1.  Full Fine-Tuning (16-bit)
        2.  LoRA Fine-Tuning (16-bit)
        3.  LoRA Fine-Tuning (8-bit)
        4.  QLoRA Fine-Tuning (4-bit)

* **Phase 2: Evaluate Proposed Method (SAPLING)**
    * **Objective:** To implement and apply the Layer Dropping framework using two different strategies and compare their performance against the baselines.
    * **Step 1 - Diagnosis (Shared):** Train LoRA adapters on all layers for 1 epoch to rank layer importance.
    * **Step 2 - Pruning Strategy A (Batched Dropping):** Prune the N least important layers at once (for N = 4, 8, 12, 16) and then fine-tune the resulting smaller model to recover performance.
    * **Step 3 - Pruning Strategy B (Successive Dropping):** Iteratively train for 1 epoch and then prune the single least important layer. This process is repeated, with early stopping implemented to halt the process if performance degrades.

<br>

## 4. Final Results

This table summarizes the final accuracy of all baseline and proposed models on the FinQA dataset. The results for Successive Dropping will be added upon completion.

| Method | Precision | Accuracy (%) | Model Size (Layers) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Full Fine-Tuning** | 16-bit (BF16) | **1.59%** | 32/32 (100%) | Performance Ceiling |
| **LoRA** | 16-bit (FP16) | 1.47% | 32/32 (100%) | High-quality Efficiency Baseline |
| **QLoRA** | 4-bit | 1.02% | 32/32 (100%) | Best Quantization Baseline |
| **LoRA** | 8-bit | 0.91% | 32/32 (100%) | Mid-quality Efficiency Baseline |
| **SAPLING (Batched, -4L)** | 16-bit (BF16) | 0.68% | 28/32 (87.5%) | 12.5% Model Size Reduction |
| **SAPLING (Batched, -8L)** | 16-bit (BF16) | 0.45% | 24/32 (75%) | 25% Model Size Reduction |
| **SAPLING (Batched, -12L)** | 16-bit (BF16) | 0.57% | 20/32 (62.5%) | 37.5% Model Size Reduction |
| **SAPLING (Batched, -16L)** | 16-bit (BF16) | 0.23% | 16/32 (50%) | 50% Model Size Reduction |
| **SAPLING (Successive)** | 16-bit (BF16) | TBD | TBD | Compares gradual vs. batched |

<br>

## 5. Setup & How to Run

This research can be reproduced using the following steps.

**1. Environment Setup**

```bash
# Clone the repository
git clone [https://github.com/DahunHan/Financial-LLM-Optimization.git](https://github.com/DahunHan/Financial-LLM-Optimization.git)
cd Financial-LLM-Optimization

# Create and activate the Python virtual environment (Python 3.10 required)
python3.10 -m venv .venv
source .venv/bin/activate # on Linux/macOS
# .\.venv\Scripts\activate # on Windows

# Install dependencies from the final, stable requirements file
pip install -r requirements.txt --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

**2. API Keys & Wandb**
```bash
# Create a .env file and add your Hugging Face Hub token
echo "HUGGING_FACE_HUB_TOKEN=hf_..." > .env

# Log in to Weights & Biases (optional)
wandb login
```
**3. Running Experiments**
```bash
# 1. Preprocess data (if not already done)
python3.10 preprocess_data.py

# 2. Run SAPLING Diagnosis (Required for all pruning)
python3.10 rank_layers.py

# 3. Run SAPLING Pruning (Choose one strategy)
# Strategy A: Batched Dropping
python3.10 prune_and_finetune.py --num_layers_to_drop 4

# Strategy B: Successive Dropping
python3.10 successive_pruning.py
```
## 6. Hardware Analysis & Environment Notes

Local Machine: NVIDIA RTX 3080 (10GB VRAM)

Cloud Instance: Lambda Labs A100 (40GB VRAM)

VRAM Analysis for Llama-2-7B (Experimental Results):

Full Fine-Tuning (16-bit): Impossible on RTX 3080. Requires ~28GB+ VRAM, necessitating cloud GPUs like the A100.

4-bit QLoRA Fine-Tuning: Most efficient method for local fine-tuning. The quantized base model fits comfortably within 10GB VRAM.

These experimentally verified trade-offs are central to the project, highlighting the need for advanced compression techniques like SAPLING to enable high-performance model training on more accessible hardware.