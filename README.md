# Financial LLM Optimization via Structural Compression

A research project on enhancing the efficiency and accuracy of Large Language Models (LLMs) for the financial domain. This project systematically evaluates various fine-tuning and compression strategies to identify optimal approaches under realistic data constraints.

<br>

## 1. Project Goal & Core Hypothesis

The goal of this research is to identify and validate the most effective optimization strategies for LLMs in the specialized financial domain. We aim to maximize the trade-off between **model performance (accuracy)** and **inference efficiency** (size, speed) on both consumer and cloud hardware.

Our core hypothesis evolved throughout the research:
> Initially, we hypothesized that **Structural Compression (SAPLING)** would outperform standard quantization. However, early experiments revealed severe **overfitting** as the primary challenge. The revised focus is to test which fine-tuning methodology offers the best performance and resistance to overfitting when trained on a diverse, combined dataset.

<br>

## 2. Methodology & Experimental Plan

This research was conducted in iterative phases, with findings from each phase informing the next.

* **Phase 1: Initial Baselines (on FinQA dataset)**
    * **Models:** Full Fine-Tuning (FFT), LoRA (16/8-bit), QLoRA (4-bit).
    * **Finding:** Severe overfitting occurred within 1-2 epochs due to the small dataset size relative to the model's capacity.

* **Phase 2: Structural Compression (on FinQA dataset)**
    * **Models:** SAPLING (Batched & Successive Layer Dropping).
    * **Finding:** Performance degraded significantly, suggesting financial reasoning is distributed across all layers, making structural pruning ineffective for this task.

* **Phase 3: Data-Centric Approaches**
    * **Experiment 3a (Augmentation):** Paraphrased the FinQA dataset using Gemini.
    * **Finding:** Performance dropped drastically (1.59% -> 0.68%), proving that data quality and diversity are more critical than sheer quantity.
    * **Experiment 3b (Combination):** Combined three distinct datasets (**FinQA**, **TAT-QA**, **FiQA**) to enhance data volume and, more importantly, diversity. This became the definitive dataset for the final experiments.

<br>

## 4. Final Results (Combined Dataset)

This table summarizes the final accuracy of all models fine-tuned on the **combined FinQA + TAT-QA + FiQA dataset**.

| Method | Precision | Accuracy (%) | Performance vs. FFT | Notes |
| :--- | :---: | :----------: | :-----------------: | :--- |
| **LoRA (16-bit)** | 16-bit (FP16) | **18.08%** | **100.1%** | **Best performing model**; demonstrates superior regularization. |
| **Full Fine-Tuning** | 16-bit (BF16) | 18.06% | 100% | New Performance Ceiling |
| **LoRA (8-bit)** | 8-bit | TBD | TBD | Currently in training. |
| **QLoRA (4-bit)** | 4-bit | TBD | TBD | Next to be trained. |

<br>

## 5. Key Findings & Future Directions

* **Data Diversity is Key:** Combining multiple datasets with different characteristics (quantitative vs. qualitative) was the single most effective strategy, boosting performance from ~1.5% to over 18%.
* **LoRA as a Superior Regularizer:** In a data-rich environment, LoRA's constrained training slightly outperformed the unconstrained Full Fine-Tuning, suggesting it's a more robust fine-tuning method that is less susceptible to noise in the training data.
* **Next Steps:** Complete the 8-bit and 4-bit LoRA experiments on the combined dataset to finalize the performance comparison of different optimization techniques.

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