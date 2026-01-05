"""
FCP (Feedback-Conditional Policy) Trainer with Ray-based single controller.

This implementation is based on the paper:
"Language Models Can Learn from Verbal Feedback Without Scalar Rewards"
Paper: https://arxiv.org/pdf/2509.22638

This trainer implements a novel training algorithm that:
1. Performs n rollouts for each prompt
2. Calls GPT API to generate critique and reward for each response
3. Restructures data as {critique}+{prompt}+{response} and performs SFT training with negative log-likelihood loss
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
from tensordict import TensorDict
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class RayFCPTrainer(RayPPOTrainer):
    """
    FCP (Feedback-Conditional Policy) Trainer.
    
    This trainer implements a novel training paradigm where:
    1. For each prompt, generate n responses via rollout
    2. Call GPT API to generate critique and reward for each response
    3. Restructure data as {critique}+{prompt}+{response} for SFT training
    4. Perform gradient updates using SFT loss
    """

    def _prepare_sft_data(self, batch: DataProto, critiques: list[str]) -> DataProto:
        """
        Prepare data for SFT training using chat template format.
        Format: user: "<critique><prompt>", assistant: "<response>"
        
        Args:
            batch: Original batch data
            critiques: List of critiques from GPT API
            
        Returns:
            DataProto: Restructured data for SFT training
        """
        sft_batch = deepcopy(batch)

        sft_batch.batch["old_input_ids"] = batch.batch["prompts"].clone()
        sft_batch.batch["old_attention_mask"] = batch.batch["attention_mask"].clone()
        sft_batch.batch["old_position_ids"] = batch.batch["position_ids"].clone()
        
        # Get original prompts and responses (decode to text first)
        batch_size = len(batch)
        new_input_ids = []
        new_attention_mask = []
        new_prompts = []
        new_position_ids = []

        prompts = batch.batch["prompts"]
        position_ids = batch.batch["position_ids"]
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]

        max_prompt_length = self.config.data.get("max_prompt_length", 1024)
        
        response = []
        for i in range(batch_size):

            item_prompt_ids = prompts[i]
            prompt_length = len(item_prompt_ids)
            item_attn_mask = attention_mask[i]
            prompt_attn_mask = item_attn_mask[:prompt_length]

            valid_prompt_ids = item_prompt_ids[prompt_attn_mask.bool()]
            prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

            user_content = prompt_text.split("</EF>")[1].split("<|im_end|>\n")[0]
            
            # Get critique text
            critique_text = critiques[i]
            
            # Construct chat format with system prompt
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{self.config.algorithm.critique_start_token}{critique_text}{self.config.algorithm.critique_end_token}{user_content}"},
            ]
            
            # Apply chat template
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            item_input_ids = model_inputs.pop("input_ids")
            item_attention_mask = model_inputs.pop("attention_mask")

            item_input_ids, item_attention_mask = verl_F.postprocess_data(
                input_ids=item_input_ids,
                attention_mask=item_attention_mask,
                max_length=max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )

            item_position_ids = compute_position_id_with_mask(item_attention_mask)

            response_ids = input_ids[i][max_prompt_length:]
            response.append(response_ids)

            new_input_ids.append(item_input_ids)
            new_attention_mask.append(item_attention_mask)
            new_prompts.append(item_prompt_ids)
            new_position_ids.append(item_position_ids)

        new_input_ids = torch.stack([x.squeeze(0) for x in new_input_ids], dim=0).to(input_ids[0].device)
        new_attention_mask = torch.stack([x.squeeze(0) for x in new_attention_mask], dim=0).to(attention_mask[0].device)
        new_prompts = torch.stack([x.squeeze(0) for x in new_prompts], dim=0).to(prompts[0].device)
        new_position_ids = torch.stack([x.squeeze(0) for x in new_position_ids], dim=0).to(position_ids[0].device)

        response = torch.stack([x.squeeze(0) for x in response], dim=0).to(input_ids[0].device)
        seq = torch.cat([new_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=new_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        response_position_ids = new_position_ids[..., -1:] + delta_position_id
        new_position_ids = torch.cat([new_position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=[151643,151645], dtype=new_attention_mask.dtype
        )
        new_attention_mask = torch.cat((new_attention_mask, response_attention_mask), dim=-1)

        sft_batch.batch["prompts"] = new_prompts
        sft_batch.batch["input_ids"] = seq
        sft_batch.batch["attention_mask"] = new_attention_mask
        sft_batch.batch["position_ids"] = new_position_ids

        return sft_batch

    def fit(self):
        """
        The main training loop for FCP algorithm.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"[FCP] Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Add progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                # Convert to DataProto
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # Prepare generation batch
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                
                # Repeat for n rollouts
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Step 1: Generate responses (n rollouts per prompt)
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Add unique IDs for tracking
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    
                    # Repeat to align with rollouts
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # Step 2: Call GPT API to generate critique and reward
                    with marked_timer("gpt_api", timing_raw, "yellow"):
                        # Compute rewards and get critiques via reward function
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True, critique_type=self.config.algorithm.critique_type, step=self.global_steps)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"[FCP] Error in reward_fn: {e}")
                            raise e

                        # Extract critiques from extra info
                        critiques = reward_extra_infos_dict.get("critiques", [])
                        
                        new_batch.batch["token_level_scores"] = reward_tensor
                        new_batch.batch["token_level_rewards"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                    # compute fsdp log_probs on old C+
                    with marked_timer("old_fsdp_log_prob", timing_raw, "blue"):
                        fsdp_old_log_prob = self.actor_rollout_wg.compute_log_prob(new_batch)
                        entropys = fsdp_old_log_prob.batch["entropys"]
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                        response_masks = new_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        fsdp_old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(fsdp_old_log_prob_metrics)
                        fsdp_old_log_prob.batch.pop("entropys")
                        new_batch = new_batch.union(fsdp_old_log_prob)
                    
                    
                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    new_batch, is_metrics = compute_rollout_correction_and_add_to_batch(new_batch, rollout_corr_config)
                    # IS and mismatch metrics already have mismatch/ prefix
                    metrics.update(is_metrics)
                    

                    # Step 3: Prepare data for SFT training
                    with marked_timer("prepare_sft", timing_raw, "blue"):
                        # Prepare SFT data with critique + prompt + response
                        sft_batch = self._prepare_sft_data(new_batch, critiques)
                        sft_batch.batch["response_mask"] = compute_response_mask(sft_batch)
                        
                        # Balance batch if configured
                        if self.config.trainer.balance_batch:
                            self._balance_batch(sft_batch, metrics=metrics)

                    # Step 4: Perform SFT training
                    with marked_timer("adv", timing_raw, "brown"):
                        sft_batch = compute_advantage(
                            sft_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    sft_batch.meta_info["global_token_num"] = torch.sum(sft_batch.batch["attention_mask"], dim=-1).tolist()

                    # update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(sft_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Save checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                # Collect metrics
                metrics.update(compute_data_metrics(batch=sft_batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=sft_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=sft_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # Clear timing

                # Add FCP specific metrics
                metrics.update({
                    "fcp/global_step": self.global_steps,
                    "fcp/n_rollouts": self.config.actor_rollout_ref.rollout.n,
                    "fcp/critiques_generated": len(critiques),
                })

                # Log metrics
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"[FCP] Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

        # Save final checkpoint if not exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
