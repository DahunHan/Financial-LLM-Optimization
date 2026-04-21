# Navigating the Accuracy-Efficiency Trade-off in Financial LLMs
### A Data-Centric Approach

**Author:** Dahun Han
**Institution:** Korea University — SW AI 융합대학원, 빅데이터 융합학과
**Advisor:** Prof. 김영근 (Youngkeun Kim)
**Date:** October 2025
**Thesis PDF:** [`Navigating the Accuracy-Efficiency Trade-off in Financial LLMs A Data-Centric Approach_한다헌.pdf`](./Navigating%20the%20Accuracy-Efficiency%20Trade-off%20in%20Financial%20LLMs%20A%20Data-Centric%20Approach_%ED%95%9C%EB%8B%A4%ED%97%8C.pdf)

---

## TL;DR

Fine-tuning Large Language Models for finance fails in a specific, costly way: high-capacity models over-specialize on narrow financial datasets and collapse in generalization. This thesis shows that **data diversity — not model-centric tuning — is the primary lever for recovery**, and then maps the accuracy–efficiency trade-off across the four leading optimization strategies, producing a **goal-dependent selection framework** for practical deployment.

| If your goal is... | Use | Why |
|---|---|---|
| Maximum accuracy | **16-bit LoRA** | 18.08% accuracy, slightly beats full fine-tuning via regularization |
| Balance of accuracy + efficiency | **Progressive Layer Dropping** | Retains ~90% of baseline performance at **18.75%** smaller model depth |
| Maximum compression | **4-bit QLoRA (NF4)** | ~75% memory footprint reduction for on-device / privacy-sensitive deployments |

---

## Motivation

The financial domain punishes model error three times over: regulatory risk, cost, and latency. Generic LLMs lack domain knowledge; fine-tuning is not optional. But the public financial datasets available for fine-tuning are narrow — each targets a specific task (numerical QA, hybrid table-text reasoning, or sentiment). Train a 7B-parameter model on one of them, and it overfits. Hard.

This project started from a concrete failure: Llama-2-7B fine-tuned on FinQA alone stagnated at **1.6% execution accuracy** on a held-out set. That failure reframed the research question from *"which optimization technique is best?"* to *"what does the training distribution need to look like before optimization becomes the real choice?"*

---

## Methodology

### 1. Data-centric foundation

A diversified training corpus was engineered by combining three heterogeneous financial datasets, each contributing a distinct reasoning modality:

- **FinQA** — numerical reasoning over financial tables and text.
- **TAT-QA** — hybrid reasoning over tables + surrounding paragraphs in real-world reports.
- **FiQA** — qualitative reasoning: opinion QA and sentiment from news, microblogs, and forums.

### 2. Comparative optimization study

Using the stabilized diversified corpus as a baseline, four optimization strategies were implemented and evaluated against full fine-tuning of Llama-2-7B:

- Full Fine-Tuning (baseline)
- 16-bit LoRA
- 4-bit QLoRA (NF4) and 8-bit LoRA
- Progressive Layer Dropping — the six least-important initial layers removed post-fine-tune, based on layer-importance probing

All models were trained on the same augmented corpus and evaluated by execution accuracy on the held-out FinQA test set.

---

## Results

**Table 1 — Comparative performance of optimization strategies.**

| Method | Final Accuracy | vs. Baseline | Size Reduction |
|---|---:|---:|---:|
| FinQA-only (initial attempt) | 1.60% | 8.9% | 0% |
| Full Fine-Tuning (baseline) | 18.06% | 100% | 0% |
| **16-bit LoRA** | **18.08%** | 100.1% | — |
| Progressive Layer Dropping (6 layers) | 16.23% | 89.9% | **18.75%** |
| 4-bit QLoRA | 9.13% | 50.5% | ~75% |
| 8-bit LoRA | 8.96% | 49.6% | ~50% |

### Three findings

1. **Data diversity is paramount.** The 10×+ accuracy jump (1.6% → 18.06%) from the diversified corpus alone — before any model-centric tuning — established that under-diversity, not under-optimization, was the real bottleneck.

2. **Optimization is goal-dependent, not universal.** 16-bit LoRA is the accuracy ceiling (its regularization effect slightly beats full fine-tuning). Progressive Layer Dropping is the balance point for enterprise latency-sensitive workloads. 4-bit QLoRA is the compression winner for on-device or privacy-constrained deployment.

3. **Quantization *method* beats bit depth.** 4-bit QLoRA (NF4) outperformed 8-bit LoRA (9.13% vs. 8.96%). Information-theoretic precision — placing quantization bins to cover equal probability mass of the weight distribution — dominates naive integer precision.

### Practical framing for finance

- **Real-time fraud detection, algorithmic trading** → 16-bit LoRA (fractional gains matter).
- **Internal market analysis, automated credit risk reports** → Progressive Layer Dropping (responsiveness + accuracy balance).
- **On-device personalized banking, privacy-sensitive local inference** → 4-bit QLoRA (footprint matters most).

---

## Repository Structure

```
.
├── Navigating the Accuracy-Efficiency Trade-off in Financial LLMs
│   A Data-Centric Approach_한다헌.pdf   # Full thesis
├── README.md
├── requirements.txt
│
├── load_finqa.py                        # FinQA dataset loader
├── combine_dev_datasets.py              # Corpus augmentation (FinQA + TAT-QA + FiQA)
├── create_test_subset.py                # Held-out test set construction
├── debug_data.py
│
├── prune_and_finetune.py                # Main fine-tuning entrypoint
├── successive_pruning.py                # Progressive Layer Dropping implementation
├── successive_pruning_nostop.py         # PLD variant (no early stopping)
├── rank_layers.py                       # Layer importance probing
├── resume_pruning.py                    # Checkpoint resume utility
├── layer_importance_ranking.json        # Output: per-layer importance scores
│
├── inference.py                         # Inference / evaluation harness
├── app.py                               # Demo app
├── clear_cache.py                       # HF cache cleanup
│
├── data/                                # (gitignored) raw + processed datasets
├── eval_results/                        # Evaluation outputs
├── evaluate/                            # Evaluation scripts
├── preprocess/                          # Dataset preprocessing utilities
├── rag/                                 # RAG experiments
├── train/                               # Training artifacts
└── wandb/                               # W&B run logs
```

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Hardware: experiments were run on Llama-2-7B (base model) — expect ≥24 GB VRAM for 16-bit LoRA / full fine-tune, ≥8 GB for 4-bit QLoRA.

### Reproducing the data-centric baseline

```bash
# 1. Build the augmented corpus (FinQA + TAT-QA + FiQA)
python combine_dev_datasets.py

# 2. Run full fine-tuning baseline
python prune_and_finetune.py --mode full_finetune

# 3. Evaluate
python inference.py --checkpoint <path>
```

### Running the optimization strategies

```bash
# 16-bit LoRA
python prune_and_finetune.py --mode lora --bits 16

# 4-bit QLoRA (NF4)
python prune_and_finetune.py --mode qlora --bits 4

# Progressive Layer Dropping (6 layers, importance-ranked)
python rank_layers.py                     # produces layer_importance_ranking.json
python successive_pruning.py --drop 6
```

(Exact flags may differ — see each script's argparse block.)

---

## Limitations & Future Work

- All experiments used a **single base model (Llama-2-7B)**. Generalization to larger or architecturally different models (e.g., Mixture-of-Experts, Mamba) is unverified.
- Progressive Layer Dropping used a **static, pre-determined** number of removed layers. An adaptive strategy that drops layers dynamically during fine-tuning — gated on layer-importance score evolution — is a natural next step.
- **Combining QLoRA with layer-pruning** (apply QLoRA on top of a layer-pruned model) was not evaluated and is a promising direction for pushing the efficiency frontier further.

---

## Citation

If this work is useful in your research, please cite:

```bibtex
@mastersthesis{han2025financialllm,
  title  = {Navigating the Accuracy-Efficiency Trade-off in Financial LLMs:
            A Data-Centric Approach},
  author = {Han, Dahun},
  school = {Korea University, Graduate School of Software and AI Convergence,
            Department of Big Data Convergence},
  year   = {2025},
  month  = {October},
  note   = {Advisor: Prof. Youngkeun Kim}
}
```

---

## Contact

**Dahun Han** — Data Specialist & Strategist, i-ESG
LinkedIn: [linkedin.com/in/dahunhan](https://www.linkedin.com/in/dahunhan/)
GitHub: [@DahunHan](https://github.com/DahunHan)

Interested in the commercialization of domain-specific AI, financial LLM optimization, and ESG data products.
