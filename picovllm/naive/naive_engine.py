import logging


class NaiveEngine():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: list[str]) -> list[str]:
        return [self._generate_one(prompt) for prompt in prompts]
    
    def _generate_one(self, prompt: str) -> str:
        msg = {"role": "user", "content": prompt}
        text = self.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=10240,
            do_sample=False
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
