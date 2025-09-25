# FCP (Feedback Conditional Policy)

A training framework that implements **Feedback Conditional Policy (FCP)** for aligning large language models with multi-dimensional feedback.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- verl framework
- Set your `OPENAI_API_KEY` environment variable before training

### Installation

```bash
pip install -r requirements.txt
```

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


## 📋 Requirements

- Python 3.10+
- verl framework
- OpenAI API key (set as environment variable)
- Additional dependencies listed in `requirements.txt`

## 📖 Citation

If you use this repository or datasets in your research, please cite:

```bibtex
@misc{fcp2024,
  title={Feedback Conditional Policy for Large Language Model Alignment},
  author={[Author Names]},
  year={2024},
  note={Code and datasets available at: https://github.com/[repository-url]}
}
```

## 📄 License

[License information to be added]

## 🤝 Contributing

[Contributing guidelines to be added]