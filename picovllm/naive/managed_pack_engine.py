import torch
from transformers.cache_utils import DynamicCache


class ManagedPackEngine():
    '''
    Packing/batching different sequences (prompts) into one (batch_size=1).
    Split the KV Cache per sequence, so that later it can be concatted together for "continous" packing.
    It costs lots of I/O for the KV Cache split and concat, but it's easy for demo the "Continous Batching" without CUDA.
    '''
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]

    @torch.inference_mode()
    def generate(self, prompts: list[str]) -> list[str]:
        device = self.model.device
        num_prompts = len(prompts)
        min_dtype = torch.finfo(self.model.dtype).min
        
        # 1. Tokenize and Prepare individual sequences
        seq_input_ids = []
        for p in prompts:
            msg = {"role": "user", "content": p}
            text = self.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
            seq_input_ids.append(ids)
        
        # 2. Prefill individually to get separate KV caches
        packed_input_ids = torch.cat(seq_input_ids).unsqueeze(0).to(device)
        lengths = [len(ids) for ids in seq_input_ids]
        offsets = [0]
        for l in lengths[:-1]:
            offsets.append(offsets[-1] + l)
        total_prefill_len = sum(lengths)

        position_ids = torch.cat([torch.arange(l) for l in lengths]).unsqueeze(0).to(device)
        attention_mask = torch.zeros((1, 1, total_prefill_len, total_prefill_len), device=device, dtype=self.model.dtype)
        for i in range(num_prompts):
            block = torch.tril(torch.ones((lengths[i], lengths[i]), device=device, dtype=self.model.dtype))
            attention_mask[0, 0, offsets[i]:offsets[i]+lengths[i], offsets[i]:offsets[i]+lengths[i]] = block
        inverted_mask = (1.0 - attention_mask) * min_dtype

        outputs = self.model(
            input_ids=packed_input_ids,
            attention_mask=inverted_mask,
            position_ids=position_ids,
            use_cache=True
        )
        
        # SPLIT the KV cache into per-sequence DynamicCache
        full_cache = outputs.past_key_values
        seq_caches = [DynamicCache() for _ in range(num_prompts)]
        num_layers = len(full_cache.layers)
        for layer_idx in range(num_layers):
            k = full_cache.layers[layer_idx].keys
            v = full_cache.layers[layer_idx].values
            for i in range(num_prompts):
                seq_caches[i].update(
                    k[:, :, offsets[i]:offsets[i]+lengths[i], :],
                    v[:, :, offsets[i]:offsets[i]+lengths[i], :],
                    layer_idx
                )

        # 3. Decoding Loop with Managed Caches
        generated_ids = [[] for _ in range(num_prompts)]
        finished = torch.zeros(num_prompts, dtype=torch.bool, device=device)
        curr_lengths = list(lengths)
        
        last_token_indices = [offsets[i] + lengths[i] - 1 for i in range(num_prompts)]
        next_tokens = torch.argmax(outputs.logits[0, last_token_indices, :], dim=-1)

        step = 0
        while not finished.all():
            for i in range(num_prompts):
                if not finished[i]:
                    token = next_tokens[i].item()
                    generated_ids[i].append(token)
                    if token in self.stop_tokens:
                        finished[i] = True
            
            if finished.all():
                break

            # Pack active next tokens
            current_input_ids = next_tokens.unsqueeze(0) # (1, num_prompts)
            current_pos_ids = torch.tensor([curr_lengths], device=device) # (1, num_prompts)
            
            # MERGE caches for the model call
            merged_cache = DynamicCache()
            for layer_idx in range(num_layers):
                k = torch.cat([c.layers[layer_idx].keys for c in seq_caches], dim=2)
                v = torch.cat([c.layers[layer_idx].values for c in seq_caches], dim=2)
                merged_cache.update(k, v, layer_idx)
            
            total_history_len = sum(curr_lengths)
            total_cache_len = total_history_len + num_prompts
            new_mask = torch.zeros((1, 1, num_prompts, total_cache_len), device=device, dtype=self.model.dtype)
            
            start = 0
            for i in range(num_prompts):
                # History
                new_mask[0, 0, i, start : start + curr_lengths[i]] = 1.0
                # Current token (the one being added)
                new_mask[0, 0, i, total_history_len + i] = 1.0
                start += curr_lengths[i]
            
            inverted_mask = (1.0 - new_mask) * min_dtype
            
            outputs = self.model(
                input_ids=current_input_ids,
                attention_mask=inverted_mask,
                position_ids=current_pos_ids,
                past_key_values=merged_cache,
                use_cache=True
            )
            
            # UPDATE sequence caches with the new tokens
            for layer_idx in range(num_layers):
                full_k = merged_cache.layers[layer_idx].keys
                full_v = merged_cache.layers[layer_idx].values
                for i in range(num_prompts):
                    # The new token for sequence i is at index (total_history_len + i)
                    new_k = full_k[:, :, total_history_len + i : total_history_len + i + 1, :]
                    new_v = full_v[:, :, total_history_len + i : total_history_len + i + 1, :]
                    seq_caches[i].update(new_k, new_v, layer_idx)

            for i in range(num_prompts):
                curr_lengths[i] += 1
            
            next_tokens = torch.argmax(outputs.logits[0, :, :], dim=-1) # (num_prompts,)

        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
