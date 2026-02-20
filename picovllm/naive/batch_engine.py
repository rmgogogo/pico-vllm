class NaiveBatchEngine():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: list[str]) -> list[str]:
        msgs = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for msg in msgs
        ]
        model_inputs = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=10240,
            do_sample=False
        )
        result = [
            self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            for i in range(len(generated_ids))
        ]
        return result
