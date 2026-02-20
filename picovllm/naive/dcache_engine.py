import torch

class DynamicCacheEngine():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_tokens = self.tokenizer.eos_token_id if isinstance(self.tokenizer.eos_token_id, list) else [self.tokenizer.eos_token_id]
    
    def generate(self, prompts: list[str]) -> list[str]:
        return [self._generate_one_dynamic_cache(prompt) for prompt in prompts]

    @torch.inference_mode()
    def _generate_one_dynamic_cache(self, prompt: str) -> str:
        msg = {"role": "user", "content": prompt}
        text = self.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        generated_ids = []
        past_key_values = None
        token_id = None
        while token_id is None or token_id not in self.stop_tokens:
            position_ids = None
            if past_key_values is not None:
                position = attention_mask.shape[1] - 1
                position_ids = torch.tensor([[position]], device=self.model.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            input_ids = next_token_id
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.model.device, dtype=attention_mask.dtype)], dim=-1)
            
            token_id = next_token_id.item()
            generated_ids.append(token_id)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)