"""
GPT Critique Reward Manager that integrates GPT API for critique generation and reward computation.
This is a general-purpose reward manager that can be used by various training algorithms.
"""

import json
import os
import random
import time
from collections import defaultdict
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
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        reference_answer_key="reference_answer",  # 新增：reference answer 字段名
        api_key=None,
        model_name="gpt-5-nano",
        max_workers=128,
        timeout=1800,
        max_retries=3,
        critique_prompt_template=None,
        cache_dir="./cache",
        cache_filename="gpt_critique_math_cache.jsonl",
        **kwargs
    ):
        """
        Initialize GPT Critique Reward Manager.
        
        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of samples to examine/print
            compute_score: Legacy compute score function (not used)
            reward_fn_key: Key for accessing data source
            api_key: OpenAI API key (if None, will get from OPENAI_API_KEY env var)
            model_name: GPT model name to use
            max_workers: Maximum number of concurrent API calls
            timeout: Timeout for API calls
            max_retries: Maximum retries for failed API calls
            critique_prompt_template: Template for critique prompt
            cache_dir: Directory to save critique cache files
            cache_filename: Filename for critique cache
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reference_answer_key = reference_answer_key  # 新增：存储 reference answer 字段名
        self.debug_mode = kwargs.get("debug_mode", True)
        
        # GPT API configuration - get from environment variable if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide api_key parameter.")
        
        self.model_name = model_name
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        
        if self.debug_mode:
            print(f"[DEBUG] GPT Critique Manager initialized with debug mode")
            print(f"[DEBUG] Model: {self.model_name}")
            print(f"[DEBUG] Max workers: {self.max_workers}")
            print(f"[DEBUG] Timeout: {self.timeout}s")
            print(f"[DEBUG] Max retries: {self.max_retries}")
        
        # Note: We will create individual clients for each thread to avoid lock contention
        
        # Use provided critique prompt template or default
        self.critique_prompt_template = critique_prompt_template or self._get_default_critique_template()
        
        # Cache configuration
        self.cache_dir = Path(cache_dir)
        self.cache_filename = cache_filename
        self.cache_path = self.cache_dir / self.cache_filename
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[GPT Critique] Initialized reward manager with model: {self.model_name}")
        print(f"[GPT Critique] Max workers: {self.max_workers}, Timeout: {self.timeout}s")
        print(f"[GPT Critique] API key found: {self.api_key[:10]}..." if self.api_key else "[GPT Critique] No API key")
        print(f"[GPT Critique] Cache path: {self.cache_path}")

    def _get_default_critique_template(self) -> str:
        """Get default critique prompt template."""
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

    def _call_gpt_api_single(self, prompt: str, response: str, sample_id: int, reference_answer: str = "") -> Dict[str, Any]:
        """
        Call GPT API for a single prompt-response pair.
        Based on the reference implementation using responses.create API.
        
        Args:
            prompt: The input prompt (question)
            response: The AI response to critique (model_answer)
            sample_id: Sample ID for logging
            reference_answer: Reference answer for correctness check (optional)
            
        Returns:
            Dict containing critique data and usage info
        """
        # Create individual OpenAI client for this thread to avoid lock contention
        client = OpenAI(api_key=self.api_key)
        
        # Format the critique prompt using the new template
        critique_prompt = self.critique_prompt_template.format(
            question=prompt.strip(),
            model_answer=response.strip(),
            reference_answer=reference_answer.strip() if reference_answer else "Not provided"
        )
        
        for attempt in range(self.max_retries):
            try:
                # Call GPT API using responses.create (following reference implementation)
                response_obj = client.responses.create(
                    model=self.model_name,
                    input=critique_prompt,
                    reasoning={
                        "effort": "minimal"
                    }
                )
                
                # Extract usage information
                u = response_obj.usage
                usage_record = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "input_tokens": getattr(u, "input_tokens", None),
                    "output_tokens": getattr(u, "output_tokens", None),
                    "reasoning_tokens": getattr(u, "reasoning_tokens", None),
                }
                
                # Extract response content
                content = response_obj.output_text.strip()
                
                # Parse critique using the new parsing method
                critique_data, parse_success = self._parse_critique(content)
                
                # If parsing failed, retry (don't count as API retry, but as parsing retry)
                if not parse_success:
                    print(f"[GPT Critique] Parsing failed for sample {sample_id}, attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        continue  # Retry the API call due to parsing failure
                    else:
                        print(f"[GPT Critique] Final parsing failure for sample {sample_id}")
                        # Use the failed parse result but mark as unsuccessful
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
                print(f"[GPT Critique] API call failed for sample {sample_id}, attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Add random delay to avoid all threads retrying simultaneously
                    time.sleep(2 + random.uniform(0, 2))
                else:
                    # Return default values on final failure
                    return {
                        "critique_data": {
                            "raw_response": "",
                            "user_feedback": "",
                            "analysis": "",
                            "high_level_critique": ""
                        },
                        "usage_record": "api_error",
                        "sample_id": sample_id,
                        "success": False
                    }
        return {
            "critique_data": {
                "raw_response": "",
                "user_feedback": "",
                "analysis": "",
                "high_level_critique": ""
            },
            "usage_record": "max_retries_reached",
            "sample_id": sample_id,
            "success": False
        }

    def _parse_critique(self, critique: str) -> Tuple[dict, bool]:
        """
        Parse critique response using the structured format.
        Based on the provided parsing rules.
        
        Args:
            critique: Raw GPT response content
            
        Returns:
            Tuple of (parsed_data_dict, parse_success_bool)
        """
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

        if result["high_level_critique"] == "" and "High-level summary" in critique:
            result["high_level_critique"] = critique.split("High-level summary")[1].strip().strip(":").strip()
        if result["high_level_critique"] == "" and "high-level summary" in critique:
            result["high_level_critique"] = critique.split("high-level summary")[1].strip().strip(":").strip()
            
        # Check if parsing was successful
        # Consider parsing successful if we have at least user_feedback and high_level_critique
        parse_success = (
            len(result["user_feedback"].strip()) > 0 and 
            len(result["high_level_critique"].strip()) > 0 and
            len(result["analysis"].strip()) > 0 and
            len(result["score"].strip()) > 0 and
            result["score"].isdigit()
        )
        
        return result, parse_success

    
    def _has_critique_in_data(self, data: DataProto) -> bool:
        """
        Check if critique data is available in the non_tensor_batch.
        
        Args:
            data: DataProto containing batch data
            
        Returns:
            True if critique data is available, False otherwise
        """
        # print(f"[GPT Critique] Checking if critique data is available in the non_tensor_batch: {data}")
        if not hasattr(data, 'non_tensor_batch') or data.non_tensor_batch is None:
            return False
        
        # Check first item to see if critique is available
        first_item = data[0]
        if not hasattr(first_item, 'non_tensor_batch') or first_item.non_tensor_batch is None:
            return False
        
        return 'critique' in first_item.non_tensor_batch


    def _save_to_cache(self, prompt: str, response: str, critique_data: dict, usage_record: dict, reward: float, critique_type: str, step: int = None):
        """
        Save prompt, response, and critique data to cache file.
        
        Args:
            prompt: Input prompt
            response: Model response
            critique_data: Parsed critique data
            usage_record: API usage information
            reward: Computed reward score
            critique_type: "user" or "pro"
            step: Current training step (optional)
        """
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

        cache_path = str(self.cache_path)
        if critique_type == "user":
            cache_path = cache_path.replace(".jsonl", "_user.jsonl")
        else:
            cache_path = cache_path.replace(".jsonl", "_pro.jsonl")

        try:
            # Append to cache file in jsonl format
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[GPT Critique] Warning: Failed to save to cache: {e}")

    def __call__(self, data: DataProto, return_dict: bool = True, critique_type: str = "user", step: int = None) -> torch.Tensor | dict[str, Any]:
        """
        Main entry point for reward computation.
        
        Args:
            data: DataProto containing prompt and response data
            return_dict: Whether to return additional information
            critique_type: Type of critique ("user" or "pro")
            step: Current training step (optional)
            
        Returns:
            Reward tensor or dict with reward and extra info
        """
        batch_size = len(data)
        
        if self.debug_mode:
            print(f"[DEBUG] GPT Critique Manager called with batch size: {batch_size}")
            print(f"[DEBUG] Return dict: {return_dict}")
        
        # Extract prompts, responses, and reference answers
        prompts = []
        responses = []
        reference_answers = []
        
        for i in range(batch_size):

            # —— 提取 prompt ——
            prompt_ids = data.batch["prompts"][i]
            prompt_length = len(prompt_ids)
            attn_mask = data.batch["attention_mask"][i]                   # [seq_len] 或 [1, seq_len]
            prompt_attn_mask = attn_mask[:prompt_length]

            valid_prompt_ids = prompt_ids[prompt_attn_mask.bool()]
            prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

            # if self.debug_mode:
            #     print(f"[DEBUG] Sample {i} - Prompt: {prompt_text}")

            user_content = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>\n")[0]
            if "</EF>" in user_content:
                user_content = user_content.split("</EF>")[1]

            # —— 提取 response ——
            response_ids = data.batch["responses"][i]       # [resp_len]
            response_length = len(response_ids)
            attn_resp = attn_mask[prompt_length: prompt_length + response_length]

            valid_response_ids = response_ids[attn_resp.bool()]
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            # if response_text.endswith("<|endoftext|>"):
            #     response_text = response_text[: -len("<|endoftext|>")]
            # if response_text.endswith("<|im_end|>\n"):
            #     response_text = response_text[: -len("<|im_end|>\n")]
            
            # Extract reference answer from non_tensor_batch
            ground_truth = data.non_tensor_batch["reference_answer"][i]
            
            prompts.append(user_content)
            responses.append(response_text)
            reference_answers.append(ground_truth)
            
            if self.debug_mode and i == 0:  # Debug first item
                print(f"[DEBUG] Sample {i} - Raw Prompt: {prompt_text}")
                print(f"[DEBUG] Sample {i} - Processed Prompt: {user_content}...")
                print(f"[DEBUG] Sample {i} - Response: {response_text[:50]}...")
                print(f"[DEBUG] Sample {i} - Reference Answer: {ground_truth}...")

        # Check if this is validation mode with reference answers
        is_validation = (data.meta_info.get("validate", False) and 
                        any(ref_ans.strip() for ref_ans in reference_answers))
        
    
        # Normal training mode - call GPT API
        # Check if we should use critique data directly (debug mode)
        use_data_critique = self.debug_mode and self._has_critique_in_data(data)
        
        if self.debug_mode:
            print(f"[DEBUG] use_data_critique: {use_data_critique}")
            if hasattr(data, 'non_tensor_batch') and data.non_tensor_batch is not None:
                print(f"[DEBUG] Available non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")
            else:
                print(f"[DEBUG] No non_tensor_batch available")
        

        print(f"[GPT Critique] Calling GPT API for {batch_size} samples with {self.max_workers} workers...")
        if self.debug_mode:
            print(f"[DEBUG] Sample prompt: {prompts[0][:100]}..." if prompts else "[DEBUG] No prompts")
            print(f"[DEBUG] Sample response: {responses[0][:100]}..." if responses else "[DEBUG] No responses")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all API calls
            if self.debug_mode:
                print(f"[DEBUG] Submitting {batch_size} API calls to ThreadPoolExecutor...")
            futures = [
                executor.submit(self._call_gpt_api_single, prompts[i], responses[i], i, reference_answers[i])
                for i in range(batch_size)
            ]
            
            # Collect results with timeout (following reference implementation)
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout)  # 3分钟超时 (following reference)
                    results.append(result)
                except Exception as e:
                    print(f"[GPT Critique] Future failed: {e}")
                    results.append({
                        "critique_data": {
                            "raw_response": "",
                            "user_feedback": "",
                            "analysis": "",
                            "high_level_critique": "",
                            "score": "5"
                        },
                        "usage_record": None,
                        "sample_id": len(results),
                        "success": False
                    })
        
        # Sort results by sample_id to maintain order
        results.sort(key=lambda x: x["sample_id"])
        
        end_time = time.time()
        if self.debug_mode:
            print(f"[DEBUG] GPT API calls completed in {end_time - start_time:.2f} seconds")
            print(f"[DEBUG] Processing {len(results)} results...")
            success_count = sum(1 for r in results if r.get("success", False))
            print(f"[DEBUG] Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        # Extract critique data, usage records and generate default rewards
        critique_data_list = []
        usage_records = []
        rewards = []
        for i, result in enumerate(results):
            critique_data_list.append(result["critique_data"])
            usage_records.append(result.get("usage_record"))
            # TODO: Generate reward score from critique data
            # For now, use a simple heuristic based on high_level_critique content
            reward = float(int(result["critique_data"]["score"]) / 10) if result["critique_data"]["score"].isdigit() else 0.5
            
            if self.debug_mode and i == 0:  # Only debug first result to avoid spam
                print(f"[DEBUG] Sample critique data: {result['critique_data']}")
                print(f"[DEBUG] Sample reward: {reward}")
            rewards.append(reward)
            
            # Save to cache (only if successful)
            if result["success"]:  # Cache only on successful attempts
                self._save_to_cache(
                    prompt=prompts[i],
                    response=responses[i], 
                    critique_data=result["critique_data"],
                    usage_record=result.get("usage_record"),
                    reward=reward,
                    critique_type=critique_type,
                    step=step
                )
        
        api_time = time.time() - start_time
        success_count = sum(1 for r in results if r["success"])
        
        print(f"[GPT Critique] GPT API completed in {api_time:.2f}s, {success_count}/{batch_size} successful")
        
        # Create reward tensor (put reward at the last token of response)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(batch_size):
            prompt_ids = data.batch["prompts"][i]
            prompt_length = len(prompt_ids)

            attn_mask = data.batch["attention_mask"][i]   
            response_ids = data.batch["responses"][i]        # 不需要 decode，这里只要长度
            response_length = len(response_ids)
            attn_resp = attn_mask[prompt_length: prompt_length + response_length]

            valid_response_length = len(response_ids[attn_resp.bool()])  # ← int(...)
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = rewards[i]
        
        
        # Print some examples for debugging
        for i in range(min(self.num_examine, batch_size)):
            critique_data = critique_data_list[i]
            print(f"\n[GPT Critique Example {i+1}]")
            print(f"Prompt: {prompts[i][:200]}...")
            print(f"Response: {responses[i][:200]}...")
            print(f"User Feedback: {critique_data['user_feedback']}")
            print(f"Analysis: {critique_data['analysis']}")
            print(f"High-level Critique: {critique_data['high_level_critique']}")
            print(f"Reward: {rewards[i]}")
        
        # Math accuracy is computed during validation mode above
        # For non-validation mode, accuracy_scores will remain empty
        if not is_validation:
            accuracy_scores = []
        # accuracy_scores is already defined in validation mode
        
        if return_dict:
            if critique_type == "user":
                critiques = [cd["user_feedback"] for cd in critique_data_list]
            else:
                critiques = [cd["high_level_critique"] for cd in critique_data_list]

            reward_extra_info = {
                # "critiques": [cd["high_level_critique"] for cd in critique_data_list],
                "critiques": critiques,
                "critique_data": critique_data_list,
                "rewards": rewards,
                "usage_records": usage_records,
                "api_success_rate": [success_count / batch_size] * batch_size,
                "api_time": [api_time] * batch_size,
            }
            
            # Add accuracy scores if computed
            if accuracy_scores:
                reward_extra_info["acc"] = accuracy_scores  # This will be treated as core metric
                reward_extra_info["math_accuracy"] = accuracy_scores  # Additional reference
            
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor
