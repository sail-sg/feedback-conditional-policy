"""
GPT Critique Reward Manager for verbal feedback-based training.

Based on: "Language Models Can Learn from Verbal Feedback Without Scalar Rewards"
Paper: https://arxiv.org/pdf/2509.22638
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("gpt_critique_math_score")
class GPTCritiqueMathScoreRewardManager(AbstractRewardManager):
    """
    GPT Critique Reward Manager that calls GPT API to generate critique and reward for each response.
    
    This manager:
    1. Extracts prompt and response from each sample
    2. Calls GPT API to generate critique and reward
    3. Returns both critique (as extra info) and reward (as tensor)
    
    This is a general-purpose manager that can be used by different training algorithms.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine=2,
        reward_fn_key="data_source",
        reference_answer_key="reference_answer",
        model_name="gpt-5-nano",
        max_workers=128,
        timeout=1800,
        max_retries=3,
        critique_prompt_template=None,
        cache_dir=None,  # User should provide cache directory
        cache_filename="gpt_critique_math_cache.jsonl",
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reference_answer_key = reference_answer_key
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.critique_prompt_template = critique_prompt_template or self._get_default_template()
        
        # User must provide cache_dir or it will use current directory
        if cache_dir is None:
            cache_dir = "./verl_cache"
            print(f"[GPT Critique] Warning: cache_dir not provided, using default: {cache_dir}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_filename = cache_filename
        self.cache_path = self.cache_dir / self.cache_filename
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[GPT Critique] Model: {self.model_name}, Workers: {self.max_workers}, Cache: {self.cache_path}")

    def _get_default_template(self) -> str:
        return """You are acting as a real-world human user of an LLM.

Inputs:
Question:
\"\"\" 
{question}
\"\"\"

Model Answer:
\"\"\" 
{model_answer}
\"\"\"

Reference Final Answer (used only for correctness check):
\"\"\" 
{reference_answer}
\"\"\"

Your tasks:

1) Simulate "user feedback" from a normal, real-world user reacting to the Model Answer only.
   - Length: 1-3 sentences, colloquial tone, first person.
   - Content: purely subjective sentiment (e.g., helpfulness, confidence, confusion, satisfaction).
   - STRICT: Do NOT mention or allude to any symbols, formulas, variable names, or specialized concepts from the Question or the Model Answer. Do NOT quote text from the inputs.

For example:
"I think you are right, but your solution is really long and complicated."
"You are a genius! You have all my respect."
"I am confused. There seems to be a mistake in your solution."
"What are you talking about? You are not answering my question."
etc.

2) Simulate a professional reviewer evaluating the Model Answer along several dimensions, including but not limited to:
   • correctness — Compare the Model Answer's final result ONLY against the Reference Final Answer (if provided). Judge whether the end result matches; do not use the reference for any other purpose.
   • logical_rigor — Assess the soundness and gaplessness of reasoning within the Model Answer itself. Do NOT use the Reference Final Answer here.
   • completeness — Judge coverage of required parts and edge cases based on the Question and the Model Answer only. Do NOT use the Reference Final Answer here.
   • clarity — Evaluate organization, readability, and ease of following in the Model Answer. Do NOT use the Reference Final Answer here.
   
Then provide a high-level summary (1-3 sentences) with overall judgment and broad observations.
- STRICT for the high-level summary: Only use adjectives and adverbs to describe the Model Answer and reasoning process. DO NOT mention where it goes wrong and where it can do better.

For example:
"Your final answer is correct, but the solution is too long and complicated. There are also several logical errors in your solution."
"The answer is partially correct. The reasoning is sound but not complete. Also, you are being too verbose."
"The answer is totally wrong. It lacks soundness and is not complete. However, the solution is concise and clear."

Hard constraints:
- Keep all content in English.
- Do not mention anything like "reference" or "python snippet".

Output format:
### User-style feedback: <your 1-3 sentence feedback>
### Analysis along several dimensions: <your 1-3 sentence analysis>
### High-level summary: <your 1-3 sentence summary>
### Score (0-10): <one overall integer score>"""

    def _default_critique_result(self, sample_id: int, reason: str = "unknown") -> Dict[str, Any]:
        return {
            "critique_data": {
                "raw_response": "",
                "user_feedback": "",
                "analysis": "",
                "high_level_critique": "",
                "score": "5"
            },
            "usage_record": reason,
            "sample_id": sample_id,
            "success": False
        }

    def _call_gpt_api_single(self, prompt: str, response: str, sample_id: int, reference_answer: str = "") -> Dict[str, Any]:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        critique_prompt = self.critique_prompt_template.format(
            question=prompt.strip(),
            model_answer=response.strip(),
            reference_answer=reference_answer.strip() if reference_answer else "Not provided"
        )
        
        for attempt in range(self.max_retries):
            try:
                response_obj = client.responses.create(
                    model=self.model_name,
                    input=critique_prompt,
                    reasoning={"effort": "minimal"}
                )
                
                u = response_obj.usage
                usage_record = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                    "reasoning_tokens": getattr(u, "reasoning_tokens", None),
                }
                
                content = response_obj.output_text.strip()
                critique_data, parse_success = self._parse_critique(content)
                
                if not parse_success:
                    if attempt < self.max_retries - 1:
                        continue
                    return {
                        "critique_data": critique_data,
                        "usage_record": usage_record,
                        "sample_id": sample_id,
                        "success": False,
                        "failure_reason": "parsing_failed"
                    }
                
                return {
                    "critique_data": critique_data,
                    "usage_record": usage_record,
                    "sample_id": sample_id,
                    "success": True
                }
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 + random.uniform(0, 2))
                else:
                    print(f"[GPT Critique] API failed for sample {sample_id}: {e}")
                    return self._default_critique_result(sample_id, "api_error")
        
        return self._default_critique_result(sample_id, "max_retries")

    def _parse_critique(self, critique: str) -> Tuple[dict, bool]:
        parts = critique.split("###")
        result = {
            "raw_response": critique,
            "user_feedback": "",
            "analysis": "",
            "high_level_critique": "",
            "score": ""
        }
        
        for part in parts:
            part = part.strip()
            if part.startswith("User-style feedback:"):
                result["user_feedback"] = part.replace("User-style feedback:", "").strip()
            elif part.startswith("Analysis along several dimensions:"):
                result["analysis"] = part.replace("Analysis along several dimensions:", "").strip()
            elif part.startswith("High-level summary:"):
                result["high_level_critique"] = part.replace("High-level summary:", "").strip()
            elif part.startswith("Score (0-10):"):
                result["score"] = part.replace("Score (0-10):", "").strip()
            elif part.startswith("Score:"):
                result["score"] = part.replace("Score:", "").strip()

        # Fallback parsing
        if not result["high_level_critique"] and "High-level summary" in critique:
            result["high_level_critique"] = critique.split("High-level summary")[1].strip().strip(":").strip()
        if result["high_level_critique"] == "" and "high-level summary" in critique:
            result["high_level_critique"] = critique.split("high-level summary")[1].strip().strip(":").strip()
            
        parse_success = (
            result["user_feedback"].strip() and 
            result["high_level_critique"].strip() and
            result["analysis"].strip() and
            result["score"].strip() and
            result["score"].isdigit()
        )
        
        return result, parse_success

    def _extract_prompts_and_responses(self, data: DataProto) -> Tuple[List[str], List[str], List[str]]:
        batch_size = len(data)
        prompts, responses, reference_answers = [], [], []
        
        for i in range(batch_size):
            # Extract prompt
            prompt_ids = data.batch["prompts"][i]
            attn_mask = data.batch["attention_mask"][i]
            prompt_attn_mask = attn_mask[:len(prompt_ids)]
            valid_prompt_ids = prompt_ids[prompt_attn_mask.bool()]
            prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            
            user_content = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>\n")[0]
            if "</EF>" in user_content:
                user_content = user_content.split("</EF>")[1]
            
            # Extract response
            response_ids = data.batch["responses"][i]
            response_attn_mask = attn_mask[len(prompt_ids): len(prompt_ids) + len(response_ids)]
            valid_response_ids = response_ids[response_attn_mask.bool()]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = data.non_tensor_batch["reference_answer"][i]
            
            prompts.append(user_content)
            responses.append(response_text)
            reference_answers.append(ground_truth)
        
        return prompts, responses, reference_answers

    def _save_to_cache(self, prompt: str, response: str, critique_data: dict, 
                       usage_record: dict, reward: float, critique_type: str, step: int = None):
        cache_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": prompt,
            "response": response,
            "critique_data": critique_data,
            "usage_record": usage_record,
            "reward": reward,
            "model_name": self.model_name,
            "step": step
        }

        cache_path = str(self.cache_path).replace(".jsonl", f"_{critique_type}.jsonl")
        try:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[GPT Critique] Cache save failed: {e}")

    def __call__(self, data: DataProto, return_dict: bool = True, 
                 critique_type: str = "user", step: int = None) -> torch.Tensor | dict[str, Any]:
        batch_size = len(data)
        prompts, responses, reference_answers = self._extract_prompts_and_responses(data)
        
        start_time = time.time()
        
        print(f"[GPT Critique] Calling API ({batch_size} samples, {self.max_workers} workers)")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._call_gpt_api_single, prompts[i], responses[i], i, reference_answers[i])
                for i in range(batch_size)
            ]
            results = []
            for future in futures:
                try:
                    results.append(future.result(timeout=self.timeout))
                except Exception as e:
                    print(f"[GPT Critique] Future timeout: {e}")
                    results.append(self._default_critique_result(len(results), "timeout"))
        
        results.sort(key=lambda x: x["sample_id"])
        
        # Extract critiques and compute rewards
        critique_data_list = []
        usage_records = []
        rewards = []
        
        for i, result in enumerate(results):
            critique_data_list.append(result["critique_data"])
            usage_records.append(result.get("usage_record"))
            
            score = result["critique_data"]["score"]
            reward = float(int(score) / 10) if score.isdigit() else 0.5
            rewards.append(reward)
            
            if result["success"]:
                self._save_to_cache(prompts[i], responses[i], result["critique_data"],
                                   result.get("usage_record"), reward, critique_type, step)
        
        api_time = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        print(f"[GPT Critique] Completed in {api_time:.2f}s ({success_count}/{batch_size} successful)")
        
        # Build reward tensor (reward at last valid token)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(batch_size):
            prompt_len = len(data.batch["prompts"][i])
            attn_mask = data.batch["attention_mask"][i]
            response_ids = data.batch["responses"][i]
            response_attn = attn_mask[prompt_len: prompt_len + len(response_ids)]
            valid_len = response_attn.bool().sum().item()
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = rewards[i]
        
        # Print examples
        for i in range(min(self.num_examine, batch_size)):
            cd = critique_data_list[i]
            print(f"\n[Example {i+1}]")
            print(f"Prompt: {prompts[i][:150]}...")
            print(f"Response: {responses[i][:150]}...")
            print(f"Critique: {cd['high_level_critique']}")
            print(f"Reward: {rewards[i]:.2f}")
        
        if not return_dict:
            return reward_tensor
        
        critiques = [cd["user_feedback" if critique_type == "user" else "high_level_critique"] 
                     for cd in critique_data_list]
        
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {
                "critiques": critiques,
                "critique_data": critique_data_list,
                "rewards": rewards,
                "usage_records": usage_records,
                "api_success_rate": [success_count / batch_size] * batch_size,
                "api_time": [api_time] * batch_size,
            }
        }
