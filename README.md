<div align="center">

# FCP (Feedback Conditional Policy)

[![arXiv](https://img.shields.io/badge/arXiv-2509.22638-b31b1b.svg)](https://arxiv.org/abs/2509.22638)
[![Hugging Face Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/Renjie-Ranger/feedback-conditional-policy)

</div>

This is the official repository for the paper [Language Models Can Learn from Verbal Feedback Without Scalar Rewards](https://arxiv.org/pdf/2509.22638).

A training framework that implements **Feedback Conditional Policy (FCP)** for aligning large language models with verbal feedback.

## 📝 Updates

- **2026-01-05**: Simplified the codebase, provided modification documentation ([MODIFICATIONS_FCP.md](./verl/MODIFICATIONS_FCP.md)), and released model checkpoints on [Hugging Face](https://huggingface.co/collections/Renjie-Ranger/feedback-conditional-policy).
- **2025-09-25**: Open-sourced this repository.

## 🚀 Quick Start

### Prerequisites

- verl framework
- Set your `OPENAI_API_KEY` environment variable before training

## 🏋️ Training

### Offline FCP Training
Use LLaMA-Factory's built-in SFT training code with the SFT datasets mentioned below.

### FCP Bootstrapping (Online) Training
Run the VERL training script:

```bash
./verl/recipe/fcp/run_fcp.sh
```

Configuration details can be found in `verl/recipe/fcp/config/fcp_trainer.yaml`.

## 📊 Datasets & Frameworks

We use different frameworks and datasets for different training stages:

### Offline FCP Training
**Framework**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)  
**Datasets**:
- [`RogerLos/FCP_big_math_pro_SFT`](https://huggingface.co/datasets/RogerLos/FCP_big_math_pro_SFT)
- [`RogerLos/FCP_general_reasoner_pro_SFT`](https://huggingface.co/datasets/RogerLos/FCP_general_reasoner_pro_SFT)

### FCP Bootstrapping (Online) Training
**Framework**: [verl](https://github.com/volcengine/verl)  
**Datasets**:
- [`RogerLos/FCP_big_math_pro_C-plus_no_concise`](https://huggingface.co/datasets/RogerLos/FCP_big_math_pro_C-plus_no_concise)
- [`RogerLos/FCP_general_reasoner_pro_C-plus_no_concise`](https://huggingface.co/datasets/RogerLos/FCP_general_reasoner_pro_C-plus_no_concise)

## 📖 Citation

If you find this code useful, please consider citing our paper:
```bib
@article{luo2025languagemodelslearnverbal,
      title={Language Models Can Learn from Verbal Feedback Without Scalar Rewards}, 
      author={Renjie Luo and Zichen Liu and Xiangyan Liu and Chao Du and Min Lin and Wenhu Chen and Wei Lu and Tianyu Pang},
      journal={arXiv preprint arXiv:2509.22638},
      year={2025}
}
```
