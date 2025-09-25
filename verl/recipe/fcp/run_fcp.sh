#!/usr/bin/env bash
# FCP (Feedback Conditional Policy) Training Script
# eval "$(conda shell.bash hook 2> /dev/null)"
conda activate fcp

echo "conda activate fcp success"

set -xeuo pipefail

# ================================
# Project Configuration
# ================================
project_name='FCP_Experiment'
# exp_name='FCP-v1-GPT5nano-critique-chat_250k_x4_pro_critique_v0'
exp_name='FCP-GPT5nano-big_math_summary_C+_no_concise_partial_online'

# ================================
# FCP Algorithm Specific Settings
# ================================
# Number of rollouts per prompt (n parameter in FCP algorithm)
n_rollouts=4
# Weight for SFT loss
sft_loss_weight=1.0
# Don't use KL penalty in FCP
use_kl_in_reward=False
# Debug mode for detailed logging
debug_mode=True

# ================================
# GPT API Configuration
# ================================
# OpenAI API settings (API key should be set via OPENAI_API_KEY environment variable)
gpt_model_name="gpt-5-nano"
gpt_max_workers=128
gpt_timeout=1800
gpt_max_retries=3

# ================================
# Data Configuration
# ================================
# Prompt and response length settings
max_prompt_length=1024
max_response_length=4096

# Batch size settings (smaller due to GPT API calls)
train_batch_size=2048
ppo_mini_batch_size=512
val_batch_size=512
gen_batch_size=${train_batch_size}

# ================================
# Model and Training Configuration
# ================================
# Learning rate and optimization
learning_rate=1e-6
weight_decay=0.01
warmup_steps_ratio=0.0
lr_scheduler_type="constant"

# Training epochs and checkpointing
total_epochs=1
test_freq=5
save_freq=5
val_before_train=True

# Loss aggregation mode
loss_agg_mode="token-mean"

# Critique type, "user" or "pro"
critique_type="pro"

# ================================
# Environment and Path Configuration
# ================================
# Ray configuration
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"./verl"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${WORLD_SIZE:-1}
# NNODES=1
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"./verl"}
MODEL_PATH="/path/to/your/model"
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/fcp/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"./data/train.parquet"}
# TRAIN_FILE=${TRAIN_FILE:-"./data/train_alternative.parquet"}
VAL_FILE=${VAL_FILE:-"./data/val.parquet"}
CACHE_DIR=${CACHE_DIR:-"./cache"}

# ================================
# Generation Configuration
# ================================
temperature=1.0
top_p=1.0
top_k=-1
do_sample=True

# Validation generation settings
val_temperature=0.0
val_top_p=1.0
val_do_sample=False
val_n=1

reward_manager="gpt_critique_math_score"

# ================================
# Performance and Hardware Settings
# ================================
# Sequence parallel and tensor parallel
sp_size=1
gen_tp=2

# Memory and batching
gpu_memory_utilization=0.75
offload=False

# ================================
# Environment Variables
# ================================
export HYDRA_FULL_ERROR=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=True
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN

# Check if OPENAI_API_KEY is set
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set!"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "Starting FCP training with the following configuration:"
echo "Project: ${project_name}"
echo "Experiment: ${exp_name}"
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_FILE}"
echo "Val data: ${VAL_FILE}"
echo "N rollouts: ${n_rollouts}"
echo "GPT model: ${gpt_model_name}"
echo "Cache dir: ${CACHE_DIR}"

# ================================
# Launch Training
# ================================
ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.fcp.main_fcp \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.response_key=model_response \
    data.truncation='left' \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=48 \
    data.reward_fn_key=data_source \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.gen_batch_size=${gen_batch_size} \
    data.val_batch_size=${val_batch_size} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.optim.warmup_style=${lr_scheduler_type} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${warmup_steps_ratio} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.n=${n_rollouts} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.do_sample=${do_sample} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.enforce_eager=True \
    algorithm.name=fcp \
    algorithm.n_rollouts=${n_rollouts} \
    algorithm.sft_loss_weight=${sft_loss_weight} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.debug_mode=${debug_mode} \
    algorithm.critique_type=${critique_type} \
    reward_model.reward_manager=${reward_manager} \
    reward_model.reward_kwargs.model_name="${gpt_model_name}" \
    reward_model.reward_kwargs.max_workers=${gpt_max_workers} \
    reward_model.reward_kwargs.timeout=${gpt_timeout} \
    reward_model.reward_kwargs.max_retries=${gpt_max_retries} \
    reward_model.reward_kwargs.cache_dir="${CACHE_DIR}" \
    reward_model.reward_kwargs.cache_filename="${reward_manager}_no_concise_partial_online_rerun_cache.jsonl" \
    reward_model.reward_kwargs.reference_answer_key="reference_answer" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=null \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=0

echo "FCP training completed!"
