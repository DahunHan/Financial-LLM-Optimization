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

* **Phase 3: Date-Centric Approach**
    * **Objective:** To investigate if data augmentation can mitigate the severe overfitting observed in earlier phases.
    * **Experiment:**
        1.  Use Gemini 1.5 Flash to paraphrase the training set, increasing its size by 4x.
        2.  Re-run the Full Fine-Tuning experiment on the augmented dataset.
<br>

## 4. Final Results

This table summarizes the final accuracy of all baseline and proposed models on the FinQA dataset, sorted by performance. The "Compression Rate" indicates the percentage of layers remaining, while "Performance Retention" measures the accuracy relative to the Full Fine-Tuning baseline.

| Method                          | Precision     | Accuracy (%) | Compression (Layers) | Performance Retention |
| :------------------------------ | :-----------: | :----------: | :------------------: | :-------------------: |
| **Full Fine-Tuning** | 16-bit (BF16) | **1.59%** | 100% (32/32)         | **100%** |
| **LoRA (16-bit)** | 16-bit (FP16) | 1.47%        | 100% (32/32)         | 92.5%                 |
| **SAPLING (Successive, -5L)** | 16-bit (BF16) | **1.13%** | 84.4% (27/32)        | **71.1%** |
| **QLoRA (4-bit)** | 4-bit         | 1.02%        | 100% (32/32)         | 64.2%                 |
| **LoRA (8-bit)** | 8-bit         | 0.91%        | 100% (32/32)         | 57.2%                 |
| **Full FT (Augmented Data)** | 16-bit (BF16) | 0.68%        | 100% (32/32)         | 42.8%                 |
| **SAPLING (Batched, -4L)** | 16-bit (BF16) | 0.68%        | 87.5% (28/32)        | 42.8%                 |
| **SAPLING (Batched, -12L)** | 16-bit (BF16) | 0.57%        | 62.5% (20/32)        | 35.8%                 |
| **SAPLING (Batched, -8L)** | 16-bit (BF16) | 0.45%        | 75.0% (24/32)        | 28.3%                 |
| **SAPLING (Batched, -16L)** | 16-bit (BF16) | 0.23%        | 50.0% (16/32)        | 14.5%                 |         |

<br>

## 5. Key Findings & Limitations

This research has yielded several key insights into the challenges of fine-tuning and compressing LLMs for specialized, high-complexity domains like finance.

* **Severe Overfitting due to Data Scarcity:** The primary limitation was the mismatch between the model's capacity (7B parameters) and the dataset's size (~6,600 samples). This led to rapid overfitting, often within a single epoch, across all fine-tuning methods.

* **Distributed Nature of Financial Reasoning:** Layer importance analysis revealed that all 32 layers of Llama-2-7B contribute almost equally to the FinQA task. This suggests that complex financial reasoning is a function distributed across the entire model, not localized in specific layers.

* **Ineffectiveness of Structural Pruning:** A direct consequence of the points above is that Layer Dropping (SAPLING) was not an effective compression strategy for this specific task. Removing any layer caused an irreparable loss in the model's distributed reasoning capability.

* **Failure of Naive Data Augmentation:** Augmenting the dataset by paraphrasing questions with Gemini led to a significant performance drop (from 1.59% to 0.68%). This indicates that simply increasing the quantity of data is insufficient; the augmented data likely introduced noise and failed to add new, diverse knowledge, confusing the model instead of helping it generalize.

## 6. Setup & How to Run

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

## 6. Key Findings & Limitations

This research has yielded several key insights into the challenges of fine-tuning and compressing Large Language Models for specialized, high-complexity domains like finance.

* **Rapid Overfitting due to Data/Model Mismatch:** The primary limitation of this study was the significant mismatch between the model's capacity and the dataset's size. The Llama-2-7B model, with its 7 billion parameters, possesses an immense capacity for memorization. When fine-tuned on the relatively small FinQA dataset (~6,600 training samples), the model exhibited severe and rapid overfitting, often within a single epoch. This was observed across all fine-tuning methods, including Full Fine-Tuning, LoRA, and both SAPLING strategies.

* **Distributed Nature of Financial Reasoning:** The layer importance ranking revealed that all 32 layers of the Llama-2-7B model have very similar, non-trivial importance scores for the FinQA task. This suggests that the complex, multi-step logical and numerical reasoning required for financial question-answering is not localized in a few specific layers. Instead, it appears to be a function distributed across the entire depth of the model.

* **Ineffectiveness of Structural Pruning for this Task:** A direct consequence of the points above is that structural pruning via Layer Dropping (SAPLING) was not an effective compression strategy for this specific use case. Both "Batched" and "Successive" dropping methods resulted in significant performance degradation that outweighed the benefits of reduced model size and potential speedups. The experiment suggests that when every layer contributes almost equally to a complex reasoning task, removing any of them causes an irreparable loss in capability.