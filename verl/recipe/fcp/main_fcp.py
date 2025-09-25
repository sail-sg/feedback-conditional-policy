"""
Main entry point for FCP (Feedback Conditional Policy) training.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from .fcp_ray_trainer import RayFCPTrainer


@hydra.main(config_path="config", config_name="fcp_trainer", version_base=None)
def main(config):
    run_fcp(config)


def run_fcp(config) -> None:
    """Run FCP training with Ray."""
    if not ray.is_initialized():
        # Initialize Ray cluster
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"[FCP] Ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Set up profiling if configured
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # Ensure main task is not scheduled on head node
class TaskRunner:
    def run(self, config):
        """Main training task runner."""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"[FCP] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # Print configuration
        debug_mode = config.algorithm.get("debug_mode", False)
        if debug_mode:
            print("[DEBUG] ===== FCP Debug Mode Enabled =====")
            print(f"[DEBUG] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
            print("[DEBUG] Full configuration:")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download model checkpoint
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Initialize tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)
        if debug_mode:
            print(f"[DEBUG] Tokenizer vocab size: {tokenizer.vocab_size}")
            print(f"[DEBUG] Tokenizer pad_token: {tokenizer.pad_token}")
            print(f"[DEBUG] Tokenizer eos_token: {tokenizer.eos_token}")
        
        # Verify FCP special tokens are in tokenizer vocabulary
        critique_start_token = config.algorithm.get("critique_start_token", "<EF>")
        critique_end_token = config.algorithm.get("critique_end_token", "</EF>")
        
        if debug_mode:
            print(f"[DEBUG] Verifying special tokens: {critique_start_token}, {critique_end_token}")
        
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            missing_tokens = []
            if critique_start_token not in vocab:
                missing_tokens.append(critique_start_token)
            if critique_end_token not in vocab:
                missing_tokens.append(critique_end_token)
            
            if debug_mode:
                print(f"[DEBUG] Tokenizer vocabulary size: {len(vocab)}")
                print(f"[DEBUG] {critique_start_token} in vocab: {critique_start_token in vocab}")
                print(f"[DEBUG] {critique_end_token} in vocab: {critique_end_token in vocab}")
                if critique_start_token in vocab:
                    print(f"[DEBUG] {critique_start_token} token ID: {vocab[critique_start_token]}")
                if critique_end_token in vocab:
                    print(f"[DEBUG] {critique_end_token} token ID: {vocab[critique_end_token]}")
            
            if missing_tokens:
                print(f"[FCP] Error: Special tokens {missing_tokens} not found in tokenizer vocabulary!")
                print("[FCP] Please make sure these tokens are added to the model's tokenizer.")
                print("[FCP] The training may not work correctly without these special tokens.")
            else:
                print(f"[FCP] Special tokens verified: {critique_start_token}, {critique_end_token}")

        from verl.single_controller.ray import RayWorkerGroup

        # Define worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Role worker mapping for FCP (only need actor)
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Resource pool configuration
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,  # 添加 Critic 映射以避免 KeyError
        }

        # Load FCP reward manager (GPT API based)
        if debug_mode:
            print("[DEBUG] Loading FCP reward manager (GPT API based)...")
            print(f"[DEBUG] Reward manager type: {config.reward_model.reward_manager}")
            print(f"[DEBUG] GPT model name: {config.reward_model.reward_kwargs.get('model_name', 'N/A')}")
            print(f"[DEBUG] Max workers: {config.reward_model.reward_kwargs.get('max_workers', 'N/A')}")
        
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=2,  # Print 2 examples for debugging
            debug_mode=debug_mode,  # Pass debug mode to reward manager
            **config.reward_model.get("reward_kwargs", {})
        )

        # Validation reward function (can be different from training reward manager)
        # Check if validation is disabled
        validation_disabled = (
            config.trainer.get("test_freq", 10) <= 0 and 
            not config.trainer.get("val_before_train", True)
        )
        
        if validation_disabled:
            val_reward_fn = None
            if debug_mode:
                print("[DEBUG] Validation is disabled - no validation reward manager loaded")
        elif hasattr(config, 'val_reward_model') and config.val_reward_model is not None:
            # Use separate validation reward manager configuration
            if debug_mode:
                print("[DEBUG] Loading separate validation reward manager...")
                print(f"[DEBUG] Validation reward manager type: {config.val_reward_model.reward_manager}")
            
            # Create a temporary config for validation reward manager
            from omegaconf import OmegaConf
            val_config = OmegaConf.create({
                'reward_model': {
                    'reward_manager': config.val_reward_model.reward_manager,
                    'reward_kwargs': config.val_reward_model.get('reward_kwargs', {}),
                    'sandbox_fusion': config.reward_model.get('sandbox_fusion', None)  # Copy sandbox_fusion config
                },
                'data': {
                    'reward_fn_key': config.data.reward_fn_key  # Copy reward_fn_key from main config
                },
                'custom_reward_function': config.get('custom_reward_function', {})  # Copy custom reward function config
            })
            
            # Prepare validation reward manager kwargs based on type
            val_reward_kwargs = dict(config.val_reward_model.get("reward_kwargs", {}))
            
            # Only pass debug_mode to reward managers that support it (e.g., gpt_critique)
            if config.val_reward_model.reward_manager == "gpt_critique":
                val_reward_kwargs["debug_mode"] = debug_mode
            
            val_reward_fn = load_reward_manager(
                val_config,
                tokenizer,
                num_examine=1,
                **val_reward_kwargs
            )
            
            if debug_mode:
                print(f"[DEBUG] Validation reward manager ({config.val_reward_model.reward_manager}) loaded successfully")
        else:
            # Use same reward manager as training (original behavior)
            if debug_mode:
                print("[DEBUG] Using same reward manager for validation as training")
            
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                num_examine=1,
                debug_mode=debug_mode,
                **config.reward_model.get("reward_kwargs", {})
            )
            
            if debug_mode:
                print("[DEBUG] Validation reward manager loaded successfully")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Initialize FCP trainer
        if debug_mode:
            print("[DEBUG] Initializing FCP trainer...")
            print(f"[DEBUG] N rollouts: {config.algorithm.get('n_rollouts', 4)}")
            print(f"[DEBUG] SFT loss weight: {config.algorithm.get('sft_loss_weight', 1.0)}")
            print(f"[DEBUG] Total epochs: {config.trainer.total_epochs}")
            print(f"[DEBUG] N GPUs per node: {config.trainer.n_gpus_per_node}")
            print(f"[DEBUG] N nodes: {config.trainer.nnodes}")
        
        trainer = RayFCPTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        
        if debug_mode:
            print("[DEBUG] FCP trainer initialized successfully")

        # Initialize workers and start training
        trainer.init_workers()
        print("[FCP] Starting FCP training...")
        trainer.fit()
        print("[FCP] Training completed!")


if __name__ == "__main__":
    main()
