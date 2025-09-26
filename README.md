# FCP (Feedback Conditional Policy)

A training framework that implements **Feedback Conditional Policy (FCP)** for aligning large language models with verbal feedback.

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

TBD