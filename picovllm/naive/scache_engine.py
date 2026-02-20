import logging

import torch
from transformers import StaticCache


class StaticCacheEngine():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
    
    def generate(self, prompts: list[str]) -> list[str]:
        max_new_tokens = 1024
        if not prompts:
            return []
        
        # 1. Format and tokenize all prompts
        formatted_prompts = []
        for p in prompts:
            msg = {"role": "user", "content": p}
            text = self.tokenizer.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            formatted_prompts.append(text)
            
        all_input_ids = [self.tokenizer(text, return_tensors="pt").input_ids[0] for text in formatted_prompts]
        
        # 2. Extract common prefix of token IDs
        common_prefix_ids = self._find_common_prefix(all_input_ids)
        # We leave at least one token for the suffix generation to ensure we have input for the first step
        prefix_len = max(0, len(common_prefix_ids) - 1)
        if prefix_len > 0:
            logging.info(f"Shared prefix length: {prefix_len} tokens")
        
        # 3. Initialize StaticCache
        max_prompt_len = max(len(ids) for ids in all_input_ids)
        max_cache_len = max_prompt_len + max_new_tokens
        
        past_key_values = StaticCache(
            config=self.model.config,
            batch_size=1,
            max_cache_len=max_cache_len,
            device=self.model.device,
            dtype=self.model.dtype
        )
        
        # 4. Pre-fill the shared prefix if it exists
        if prefix_len > 0:
            prefix_tensor = torch.tensor([common_prefix_ids[:prefix_len]], device=self.model.device)
            cache_position = torch.arange(prefix_len, device=self.model.device)
            # The attention mask for the prefix
            attention_mask = torch.ones((1, prefix_len), device=self.model.device, dtype=torch.long)
            
            self.model(
                input_ids=prefix_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True
            )
            
        # 5. Generate for each prompt
        results = []
        for input_ids in all_input_ids:
            suffix_ids = input_ids[prefix_len:]
            res = self._generate_from_suffix(suffix_ids, prefix_len, past_key_values, max_new_tokens)
            results.append(res)
            
        return results

    def _find_common_prefix(self, all_ids: list[torch.Tensor]) -> list[int]:
        if not all_ids:
            return []
        min_len = min(len(ids) for ids in all_ids)
        prefix = []
        for i in range(min_len):
            val = all_ids[0][i].item()
            if all(ids[i].item() == val for ids in all_ids[1:]):
                prefix.append(val)
            else:
                break
        return prefix

    @torch.inference_mode()
    def _generate_from_suffix(self, suffix_ids: torch.Tensor, prefix_len: int, past_key_values: StaticCache, max_new_tokens: int) -> str:
        device = self.model.device
        suffix_len = len(suffix_ids)
        total_prompt_len = prefix_len + suffix_len
        
        # Current input can be the suffix_ids
        input_ids = suffix_ids.unsqueeze(0).to(device)
        attention_mask = torch.ones((1, total_prompt_len), device=device, dtype=torch.long)
        cache_position = torch.arange(prefix_len, total_prompt_len, device=device)
        
        generated_ids = []
        
        # Process the suffix
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True
        )
        
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
        generated_ids.append(next_token_id.item())
        
        # Decode
        for i in range(max_new_tokens - 1):
            if generated_ids[-1] in self.stop_tokens:
                break
                
            # Update attention mask and cache position
            current_len = total_prompt_len + len(generated_ids)
            attention_mask = torch.ones((1, current_len), device=device, dtype=torch.long)
            cache_position = torch.tensor([current_len - 1], device=device)
            
            outputs = self.model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_ids.append(next_token_id.item())
            
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)