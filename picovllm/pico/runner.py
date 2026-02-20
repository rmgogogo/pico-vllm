import torch
from transformers.cache_utils import DynamicCache

from picovllm.pico.scheduler import ScheduledTask
from picovllm.pico.sequence import SequenceStatus


class Runner:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
        self.seq2kvcache = {}

    @torch.inference_mode()
    def run(self, task: ScheduledTask):
        if not task.seqs:
            return

        if task.prefill:
            next_tokens = self._prefill(task.seqs)
        else:
            next_tokens = self._decode(task.seqs)

        for i, seq in enumerate(task.seqs):
            token = next_tokens[i].item()
            seq.append_token(token)
            if token in self.stop_tokens:
                seq.status = SequenceStatus.FINISHED
                if seq.id in self.seq2kvcache:
                    del self.seq2kvcache[seq.id]
            else:
                seq.status = SequenceStatus.DECODING

    def _prefill(self, seqs):
        device = self.model.device
        dtype = self.model.dtype
        min_dtype = torch.finfo(dtype).min
        num_seqs = len(seqs)

        seq_ids = [seq.id for seq in seqs]
        seq_tensors = [torch.tensor(seq.token_ids, device=device, dtype=torch.long) for seq in seqs]
        lengths = [len(s) for s in seq_tensors]
        offsets = [0]
        for l in lengths[:-1]:
            offsets.append(offsets[-1] + l)
        total_len = sum(lengths)

        packed_input_ids = torch.cat(seq_tensors).unsqueeze(0)
        position_ids = torch.cat([torch.arange(l, device=device) for l in lengths]).unsqueeze(0)

        attention_mask = torch.zeros((1, 1, total_len, total_len), device=device, dtype=dtype)
        for i in range(num_seqs):
            block = torch.tril(torch.ones((lengths[i], lengths[i]), device=device, dtype=dtype))
            attention_mask[0, 0, offsets[i]:offsets[i]+lengths[i], offsets[i]:offsets[i]+lengths[i]] = block
        inverted_mask = (1.0 - attention_mask) * min_dtype

        outputs = self.model(
            input_ids=packed_input_ids,
            attention_mask=inverted_mask,
            position_ids=position_ids,
            use_cache=True
        )

        full_cache = outputs.past_key_values
        num_layers = len(full_cache.layers)
        for i, seq_id in enumerate(seq_ids):
            seq_cache = DynamicCache()
            for layer_idx in range(num_layers):
                k = full_cache.layers[layer_idx].keys[:, :, offsets[i]:offsets[i]+lengths[i], :]
                v = full_cache.layers[layer_idx].values[:, :, offsets[i]:offsets[i]+lengths[i], :]
                seq_cache.update(k, v, layer_idx)
            self.seq2kvcache[seq_id] = seq_cache

        last_token_indices = [offsets[i] + lengths[i] - 1 for i in range(num_seqs)]
        next_tokens = torch.argmax(outputs.logits[0, last_token_indices, :], dim=-1)
        return next_tokens

    def _decode(self, seqs):
        device = self.model.device
        dtype = self.model.dtype
        min_dtype = torch.finfo(dtype).min
        num_seqs = len(seqs)

        seq_ids = [seq.id for seq in seqs]
        curr_lengths = [self.seq2kvcache[sid].get_seq_length() for sid in seq_ids]

        current_input_ids = torch.tensor([[seq.token_ids[-1] for seq in seqs]], device=device, dtype=torch.long)
        current_pos_ids = torch.tensor([curr_lengths], device=device, dtype=torch.long)

        merged_cache = DynamicCache()
        num_layers = len(self.seq2kvcache[seq_ids[0]].layers)
        for layer_idx in range(num_layers):
            k = torch.cat([self.seq2kvcache[sid].layers[layer_idx].keys for sid in seq_ids], dim=2)
            v = torch.cat([self.seq2kvcache[sid].layers[layer_idx].values for sid in seq_ids], dim=2)
            merged_cache.update(k, v, layer_idx)

        total_history_len = sum(curr_lengths)
        total_cache_len = total_history_len + num_seqs
        new_mask = torch.zeros((1, 1, num_seqs, total_cache_len), device=device, dtype=dtype)
        
        start = 0
        for i in range(num_seqs):
            new_mask[0, 0, i, start : start + curr_lengths[i]] = 1.0
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

        for layer_idx in range(num_layers):
            full_k = merged_cache.layers[layer_idx].keys
            full_v = merged_cache.layers[layer_idx].values
            for i, seq_id in enumerate(seq_ids):
                new_k = full_k[:, :, total_history_len + i : total_history_len + i + 1, :]
                new_v = full_v[:, :, total_history_len + i : total_history_len + i + 1, :]
                self.seq2kvcache[seq_id].update(new_k, new_v, layer_idx)

        next_tokens = torch.argmax(outputs.logits[0, :, :], dim=-1)
        return next_tokens

    def _debug_naive_run(self, task: ScheduledTask):
        for seq in task.seqs:
            input_ids = torch.tensor([seq.token_ids], dtype=torch.long).to(self.model.device)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10240,
                do_sample=False,
            )
            generated_ids = output[0, seq.input_len:].tolist()
            seq.token_ids.extend(generated_ids)
            seq.status = SequenceStatus.FINISHED
