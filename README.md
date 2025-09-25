# FCP (Feedback Conditional Policy)

A training framework that implements **Feedback Conditional Policy (FCP)** for aligning large language models with verbal feedback.

## 🚀 Quick Start

### Prerequisites

- verl framework
- Set your `OPENAI_API_KEY` environment variable before training

## 🏋️ Training

Run the training script:

```bash
./verl/recipe/fcp/run_fcp.sh
```

Configuration details can be found in `verl/recipe/fcp/config/fcp_trainer.yaml`.

## 📊 Datasets

We release the training datasets on 🤗 Hugging Face Datasets:

- [`RogerLos/FCP_big_math_pro_C-plus_no_concise`](https://huggingface.co/datasets/RogerLos/FCP_big_math_pro_C-plus_no_concise)
- [`RogerLos/FCP_general_reasoner_pro_C-plus_no_concise`](https://huggingface.co/datasets/RogerLos/FCP_general_reasoner_pro_C-plus_no_concise)

## 📖 Citation

TBD