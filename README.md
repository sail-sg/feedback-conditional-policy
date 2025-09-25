# FCP (Feedback Conditional Policy)

A training framework that implements a novel algorithm:
1. Perform n rollouts for each prompt to generate responses
2. Call GPT API to generate critique and reward for each response  
3. Restructure data as {critique}+{prompt}+{response} for SFT training
4. Perform gradient updates using SFT loss

## Training

Run the training script:

```bash
./verl/recipe/fcp/run_fcp_pro_no_concise_partial_online_rerun.sh
```

## Requirements

- Set `OPENAI_API_KEY` environment variable
- Configure paths in `verl/recipe/fcp/config/fcp_trainer.yaml`
