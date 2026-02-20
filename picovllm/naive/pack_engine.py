import torch


class PackEngine():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
    
    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[str]:
        device = self.model.device
        num_prompts = len(prompts)
        min_dtype = torch.finfo(self.model.dtype).min
        
        # 1. Tokenize and Pack
        all_input_ids = []
        lengths = []
        for p in prompts:
            msg = {"role": "user", "content": p}
            text = self.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
            all_input_ids.append(ids)
            lengths.append(len(ids))
            
        input_ids = torch.cat(all_input_ids).unsqueeze(0).to(device) # (1, TotalLen)
        
        offsets = []
        curr = 0
        for l in lengths:
            offsets.append(curr)
            curr += l
        total_prefill_len = curr

        # 2. Create Initial 4D Block-Diagonal Mask and Position IDs
        # Position IDs reset for each prompt: [0, 1, 2, ..., 0, 1, ...]
        position_ids = torch.cat([torch.arange(l) for l in lengths]).unsqueeze(0).to(device)
        
        # 4D Mask: (batch, 1, query_len, key_len)
        attention_mask = torch.zeros((1, 1, total_prefill_len, total_prefill_len), device=device, dtype=self.model.dtype)
        for i in range(num_prompts):
            block = torch.tril(torch.ones((lengths[i], lengths[i]), device=device, dtype=self.model.dtype))
            attention_mask[0, 0, offsets[i]:offsets[i]+lengths[i], offsets[i]:offsets[i]+lengths[i]] = block
        
        inverted_mask = (1.0 - attention_mask) * min_dtype

        # 3. Prefill
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=inverted_mask,
            position_ids=position_ids,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        
        # 4. Decoding Loop
        generated_ids = [[] for _ in range(num_prompts)]
        finished = torch.zeros(num_prompts, dtype=torch.bool, device=device)
        curr_lengths = list(lengths)
        
        # Index of the last token for each prompt in the packed sequence
        last_token_indices = [offsets[i] + lengths[i] - 1 for i in range(num_prompts)]
        step = -1
        while not finished.all():
            step = step + 1
            # Extract logits for the last token of each prompt
            next_token_logits = outputs.logits[0, last_token_indices, :] 
            next_tokens = torch.argmax(next_token_logits, dim=-1) # (num_prompts,)
            
            # Record tokens and check for stop tokens
            for i in range(num_prompts):
                if not finished[i]:
                    token = next_tokens[i].item()
                    generated_ids[i].append(token)
                    if token in self.stop_tokens:
                        finished[i] = True
            
            if finished.all():
                break
                
            # Prepare next step inputs
            current_input_ids = next_tokens.unsqueeze(0) # (1, num_prompts)
            current_pos_ids = torch.tensor([curr_lengths], device=device) # (1, num_prompts)
            
            # Update 4D Mask for decoding step
            # The tokens just added and the ones we are about to add will be interleaved in the KV cache
            total_curr_cache_len = total_prefill_len + (step + 1) * num_prompts
            new_mask = torch.zeros((1, 1, num_prompts, total_curr_cache_len), device=device, dtype=self.model.dtype)
            
            for i in range(num_prompts):
                # Token i attends to its specific prefill block
                new_mask[0, 0, i, offsets[i] : offsets[i] + lengths[i]] = 1.0
                # Token i attends to its own previously decoded tokens (at interleaved positions)
                decoded_indices = torch.arange(step + 1, device=device) * num_prompts + total_prefill_len + i
                new_mask[0, 0, i, decoded_indices] = 1.0
            
            inverted_mask = (1.0 - new_mask) * min_dtype
            
            # Update lengths for the next step
            for i in range(num_prompts):
                curr_lengths[i] += 1

            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=inverted_mask,
                position_ids=current_pos_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            # In decoding steps, input_ids length is num_prompts, so indices are 0 to num_prompts-1
            last_token_indices = list(range(num_prompts))

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
