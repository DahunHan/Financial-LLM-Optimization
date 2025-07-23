# Financial LLM Optimization via Structural Compression

A research project on enhancing the efficiency and accuracy of Large Language Models (LLMs) for the financial domain.

<br>

## 1. Project Goal & Hypothesis

The goal of this research is to explore novel optimization strategies for LLMs specialized in the financial domain. We aim to move beyond the limitations of conventional compression techniques by maximizing **practical inference efficiency** (i.e., speed and memory usage) on **general-purpose hardware**, while maintaining the **high level of accuracy** required for complex financial tasks.

Our core hypothesis is as follows:

> The financial domain demands exceptional accuracy from LLMs. Conventional compression techniques like **Quantization** and **Pruning** can be detrimental, as they uniformly degrade model parameters, potentially compromising the critical, complex reasoning capabilities essential in finance. In contrast, we hypothesize that **Structural Compression** methods, such as **Layer Dropping**, will achieve a superior trade-off. By selectively removing entire layers (knowledge modules) deemed less relevant to the financial domain, this approach can not only secure significant, hardware-agnostic efficiency gains but also better preserve the model's core reasoning accuracy.

<br>

## 2. Methodology

To validate our hypothesis, this research will implement and compare the following methodologies.

-   **Proposed Method: SAPLING (Successive Adaptation and Compression)**
    -   **Description:** A framework that performs domain-specific adaptation (fine-tuning) and compression simultaneously. Its core technique is **Successive Layer Dropping**, which progressively reduces the model's depth by dynamically identifying and removing the least important layers after each training epoch.
    -   **Implementation:** As no official code is available, the core logic will be re-implemented based on Algorithm 1 in the original paper by Anonymous et al., 2024.

-   **Baseline 1: QLoRA (Quantized Low-Rank Adaptation)**
    -   **Description:** A Parameter-Efficient Fine-Tuning (PEFT) method that enables fine-tuning of large models in low-VRAM environments (e.g., consumer-grade GPUs). In this research, it will serve as our **performance ceiling baseline** under hardware constraints.

-   **Baseline 2: Quantization (Post-Training)**
    -   **Description:** A representative compression technique that reduces model size by converting the weights of a fully trained model to a lower precision (e.g., 4-bit). We will apply methods like `GPTQ` to measure its performance and efficiency trade-offs.

<br>

## 3. Experiments

We will conduct a systematic, three-phase experimental process to test our hypothesis.

-   **Phase 1: Establish Performance Baseline (via QLoRA Fine-Tuning)**
    -   **Objective:** To fine-tune the base LLM (Llama-2-7B) on a financial QA dataset (`FinQA`) to establish the performance upper-bound achievable under our hardware constraints.
    -   **Key Metrics:** QA Accuracy (%), Max VRAM Usage (GB), Total Training Time.

-   **Phase 2: Analyze Quantization Baseline**
    -   **Objective:** To quantify the limitations of a standard compression technique by applying post-training quantization to the model from Phase 1.
    * **Key Metrics:** Accuracy Drop (%), Model Size Reduction (%), and **practical inference throughput (tokens/sec) on general-purpose hardware**.

-   **Phase 3: Evaluate Proposed Method (SAPLING)**
    -   **Objective:** To apply our implementation of the SAPLING framework and compare its performance curve (accuracy vs. model size) against the baselines.
    * **Key Metrics:** A Pareto frontier graph of Model Size (%) vs. Accuracy (%), and the final inference throughput (tokens/sec).

<br>

## 4. Results

*This section will be updated as the experiments are completed.*

| Method                   | Accuracy (%) | Model Size (GB) | Throughput (tokens/sec) | VRAM (Inference, GB) |
| ------------------------ | :----------: | :-------------: | :---------------------: | :------------------: |
| QLoRA Fine-Tuning (FP16) |     TBD      |       TBD       |           TBD           |         TBD          |
| + GPTQ 4-bit             |     TBD      |       TBD       |           TBD           |         TBD          |
| + SAPLING (50% Drop)     |     TBD      |       TBD       |           TBD           |         TBD          |

<br>

## 5. Setup & How to Run

This research can be reproduced using the following steps.

**1. Environment Setup**
```bash
# Clone the repository
git clone [https://github.com/DahunHan/Financial-LLM-Optimization.git](https://github.com/DahunHan/Financial-LLM-Optimization.git)
cd Financial-LLM-Optimization

# Create and activate the Python virtual environment
python -m venv .venv
source .venv/bin/activate # on Linux/macOS
# .\.venv\Scripts\activate # on Windows

# Install dependencies
pip install -r requirements.txt

```
**2. API Keys**
Create a .env file in the root directory and add your Hugging Face Hub token:

```bash
HUGGING_FACE_HUB_TOKEN="hf_..."
```
**3. Running Experiments**
```bash
# Run the data preprocessing script
python preprocess_data.py

# 2. Run a training script (choose one)
python train_qlora_4bit.py
# OR
python train_lora_8bit.py
```
<br>

## 6. Hardware Constraints & Setup Notes
This research is conducted on a system with an NVIDIA RTX 3080 GPU with 10GB of VRAM. This hardware imposes significant constraints that directly inform the methodological choices of this study.

VRAM Analysis for Llama-2-7B (Experimental Results)
Full Fine-Tuning (16-bit): Theoretically Impossible. A 7-billion parameter model requires approximately 14GB of VRAM (7B parameters * 2 bytes/parameter) just to load the model weights, before accounting for optimizer states and gradients. This exceeds the available 10GB.

8-bit LoRA Fine-Tuning: Possible, but with a critical optimization. A standard 8-bit LoRA attempt initially failed with a CUDA out of memory error. However, by enabling Gradient Checkpointing, which trades a small amount of computation time for significant memory savings, the 8-bit fine-tuning was experimentally confirmed as possible on the 10GB GPU. This method is viable but noticeably slower than 4-bit QLoRA.

4-bit QLoRA Fine-Tuning: Experimentally Confirmed as the Most Efficient Method. By quantizing the base model's weights to 4-bit precision, the memory requirement for the base model is reduced to approximately 3.5GB. This provides sufficient headroom for a stable and significantly faster fine-tuning process compared to the 8-bit alternative.

Due to these experimentally verified trade-offs, 4-bit QLoRA is the foundational and recommended fine-tuning method for this project, offering the best balance of speed and resource efficiency on the given hardware.