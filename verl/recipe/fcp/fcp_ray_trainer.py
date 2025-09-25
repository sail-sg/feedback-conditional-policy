"""
FCP (Feedback Conditional Policy) Trainer with Ray-based single controller.
This trainer implements a novel training algorithm that:
1. Performs n rollouts for each prompt
2. Calls GPT API to generate critique and reward for each response
3. Restructures data as {critique}+{prompt}+{response} and performs SFT training
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
    FCP (Feedback Conditional Policy) Trainer.
    
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
        if self.config.algorithm.debug_mode:
            print("[DEBUG] ===== Preparing SFT Data =====")
            print(f"[DEBUG] Batch size: {len(batch)}")
            print(f"[DEBUG] Number of critiques: {len(critiques)}")
        
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

        print(f"[DEBUG] prompts[0]: {prompts[0]}, type: {type(prompts[0])}")
        
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
            
            if self.config.algorithm.debug_mode and i == 0:  # Only debug first item to avoid spam
                print(f"[DEBUG] ===== Sample {i} =====")
                print(f"[DEBUG] Original prompt length: {len(item_prompt_ids)}")
                print(f"[DEBUG] Valid prompt length: {prompt_length}")
                print(f"[DEBUG] Prompt text: {prompt_text}")
                print(f"[DEBUG] User content: {user_content}")
                print(f"[DEBUG] Critique text: {critique_text}")
            
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

            # if len(input_ids) > max_prompt_length:
            #     raise RuntimeError(f"Prompt length {len(input_ids)} is greater than max_prompt_length {max_prompt_length}")

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

        print(f"[DEBUG] : (after sft_batch) prompts.shape: {sft_batch.batch['prompts'].shape}, input_ids.shape: {sft_batch.batch['input_ids'].shape}, attention_mask.shape: {sft_batch.batch['attention_mask'].shape}, position_ids.shape: {sft_batch.batch['position_ids'].shape}")

        print(f"[DEBUG] (after sft_batch) prompts[0]: {new_prompts[0]}")
        print(f"[DEBUG] (after sft_batch) input_ids[0]: {seq[0]}")
        print(f"[DEBUG] (after sft_batch) attention_mask[0]: {new_attention_mask[0]}")
        print(f"[DEBUG] (after sft_batch) position_ids[0]: {new_position_ids[0]}")

        return sft_batch

    
    def _create_fake_generation_output(self, gen_batch: DataProto) -> DataProto:
        """
        Create fake generation output using model_response from data.
        This is used in debug mode to skip actual rollout.
        
        Args:
            batch: Input batch containing model_response in non_tensor_batch
            
        Returns:
            DataProto with responses field populated from model_response
        """
        idx = gen_batch.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = gen_batch.batch["attention_mask"]
        position_ids = gen_batch.batch["position_ids"]

        print(f"[DEBUG] idx shape(before fake generation): {idx.shape}")
        print(f"[DEBUG] attention_mask shape(before fake generation): {attention_mask.shape}")
        print(f"[DEBUG] position_ids shape(before fake generation): {position_ids.shape}")

        # used to construct attention_mask
        eos_token_id = self.tokenizer.eos_token_id

        batch_size = idx.size(0)

        non_tensor_batch = gen_batch.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.tokenizer.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
        
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("sharding manager is not work properly.")

        # Get model responses from data
        model_responses = gen_batch.non_tensor_batch['model_response']
        
        # Tokenize model responses
        response = []
        
        for response_text in model_responses:
            # Handle string or list inputs
            if isinstance(response_text, (list, tuple)):
                response_text = response_text[0] if len(response_text) > 0 else ""
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            response_text += "<|im_end|>\n"
            
            # Tokenize the response
            response_token_ids = self.tokenizer(
                response_text
            )["input_ids"]
            
            response.append(response_token_ids)
            
        response = pad_2d_list_to_length(response, self.tokenizer.pad_token_id, max_length=self.config.data.max_response_length).to(idx.device)

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch=TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        
        print(f"[DEBUG] prompts shapes(after fake generation): {batch['prompts'].shape}, type: {type(batch['prompts'])}")
        print(f"[DEBUG] responses shape(after fake generation): {batch['responses'].shape}, type: {type(batch['responses'])}")
        print(f"[DEBUG] idx shape(after fake generation): {batch['input_ids'].shape}, type: {type(batch['input_ids'])}")
        print(f"[DEBUG] attention_mask shape(after fake generation): {batch['attention_mask'].shape}, type: {type(batch['attention_mask'])}")
        print(f"[DEBUG] position_ids shape(after fake generation): {batch['position_ids'].shape}, type: {type(batch['position_ids'])}")
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

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
                    # non_tensor_batch_keys=["raw_prompt_ids", "model_response"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                
                # Repeat for n rollouts
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Step 1: Generate responses (n rollouts per prompt)
                    with marked_timer("gen", timing_raw, "red"):
                        print("[DEBUG] Step 1: Generating responses...")
                        print(f"[DEBUG] gen_batch non_tensor_batch keys: {gen_batch.non_tensor_batch.keys()}")
                        print(f"[DEBUG] gen_batch batch keys: {gen_batch.batch.keys()}")
                        print(f"[DEBUG] gen_batch meta_info keys: {gen_batch.meta_info.keys()}")

                        
                        # In debug mode, check if model_response is provided in data
                        if self.config.algorithm.debug_mode and hasattr(gen_batch, 'non_tensor_batch') and gen_batch.non_tensor_batch is not None and 'model_response' in gen_batch.non_tensor_batch and True:

                            print("[DEBUG] Using provided model_response from data instead of rollout")
                            # Create fake generation output using the provided model_response
                            gen_batch_output = self._create_fake_generation_output(gen_batch)
                            print(f"[DEBUG] Using {len(gen_batch_output)} model responses from data")

                        else:
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

                    print(f"[DEBUG] new_batch batch keys(after union): {new_batch.batch.keys()}")
                    print(f"[DEBUG] new_batch non_tensor_batch keys(after union): {new_batch.non_tensor_batch.keys()}")
                    print(f"[DEBUG] new_batch meta_info keys(after union): {new_batch.meta_info.keys()}")
                    print(f"[DEBUG] new_batch batch[responses][0]: {new_batch.batch['responses'][0].tolist()}, shape: {new_batch.batch['responses'][0].shape}")
                    print(f"[DEBUG] new_batch batch[attention_mask][0]: {new_batch.batch['attention_mask'][0].tolist()}, shape: {new_batch.batch['attention_mask'][0].shape}")

                    # Step 2: Call GPT API to generate critique and reward
                    with marked_timer("gpt_api", timing_raw, "yellow"):
                        if self.config.algorithm.debug_mode:
                            print("[DEBUG] Step 2: Calling GPT API for critique and reward...")
                        
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
                        
                        if self.config.algorithm.debug_mode:
                            print(f"[DEBUG] Received {len(critiques)} critiques")
                            print(f"[DEBUG] Reward tensor shape: {reward_tensor.shape}")
                            if len(critiques) > 0:
                                print(f"[DEBUG] Sample critique: {critiques[0][:100]}...")
                        
                        new_batch.batch["token_level_scores"] = reward_tensor
                        new_batch.batch["token_level_rewards"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                    # Step 3: Prepare data for SFT training
                    with marked_timer("prepare_sft", timing_raw, "blue"):
                        if self.config.algorithm.debug_mode:
                            print("[DEBUG] Step 3: Preparing SFT data...")
                        
                        # new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                        # Prepare SFT data with critique + prompt + response
                        sft_batch = self._prepare_sft_data(new_batch, critiques)
                        sft_batch.batch["response_mask"] = compute_response_mask(sft_batch)
                        
                        # Balance batch if configured
                        if self.config.trainer.balance_batch:
                            if self.config.algorithm.debug_mode:
                                print("[DEBUG] Balancing batch...")
                            self._balance_batch(sft_batch, metrics=metrics)
                        
                        if self.config.algorithm.debug_mode:
                            print(f"[DEBUG] SFT batch prepared, size: {len(sft_batch)}")

                    # Debug: Print detailed mask and ID information for the first sample
                    if self.config.algorithm.debug_mode and len(sft_batch) > 0:
                        print(f"[DEBUG] ===== Final Mask and ID Debug Information =====")
                        print(f"[DEBUG] Sample index: 0")

                        # Print input IDs
                        input_ids = sft_batch.batch["input_ids"][0]
                        print(f"[DEBUG] Input IDs (final concatenated): {input_ids.tolist()}, shape: {input_ids.shape}")

                        # Print loss mask
                        attention_mask = sft_batch.batch["attention_mask"][0]
                        print(f"[DEBUG] Attention mask: {attention_mask.tolist()}, shape: {attention_mask.shape}")
                        attention_mask_positions = [j for j, val in enumerate(attention_mask.tolist()) if val == 1]
                        print(f"[DEBUG] Attention mask non-zero positions: {attention_mask_positions}")

                        # Print response mask
                        response_mask = sft_batch.batch["response_mask"][0]
                        print(f"[DEBUG] Response mask: {response_mask.tolist()}, shape: {response_mask.shape}")
                        response_mask_positions = [j for j, val in enumerate(response_mask.tolist()) if val == 1]
                        print(f"[DEBUG] Response mask non-zero positions: {response_mask_positions}")

                        # Print responses
                        responses = sft_batch.batch["responses"][0]
                        print(f"[DEBUG] Responses: {responses.tolist()}, shape: {responses.shape}")
                    
                    # import sys
                    # sys.exit()

                    # Step 4: Perform SFT training
                    with marked_timer("adv", timing_raw, "brown"):
                        sft_batch = compute_advantage(
                            sft_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    print("[DEBUG] sft_batch.batch keys:", list(sft_batch.batch.keys()) if sft_batch.batch is not None else [])
                    print("[DEBUG] sft_batch.non_tensor_batch keys:", list(sft_batch.non_tensor_batch.keys()) if sft_batch.non_tensor_batch is not None else [])
                    print("[DEBUG] sft_batch.meta_info keys:", list(sft_batch.meta_info.keys()) if sft_batch.meta_info is not None else [])

                    sft_batch.meta_info["global_token_num"] = torch.sum(sft_batch.batch["attention_mask"], dim=-1).tolist()
                    

                    # Ensure temperature is available for actor.update_policy even if old_log_probs recomputation is skipped
                    if self.config.actor_rollout_ref.actor.policy_loss.loss_mode == "sft":
                        sft_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                        sft_batch.batch["old_log_probs"] = None
                    else:
                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(sft_batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = sft_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            sft_batch = sft_batch.union(old_log_prob)

                    # update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(sft_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                
                # import sys; sys.exit()

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

                # if self.global_steps >= 6:
                #     import sys; sys.exit()

        # Save final checkpoint if not exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
