import logging
from dataclasses import dataclass

from picovllm.pico.sequence import Sequence
from picovllm.pico.scheduler import Scheduler
from picovllm.pico.runner import Runner


@dataclass
class Config:
    max_prefill_tokens: int = 10240
    max_decode_tokens: int = 10240

class PicoEngine():
    def __init__(self, model, tokenizer, **kwargs):
        self.config = Config(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler = Scheduler(self.config)
        self.runner = Runner(self.config, model, tokenizer)
    
    def generate(self, prompts: list[str]) -> list[str]:
        # add to waiting queue
        for prompt in prompts:
            self._add_request(prompt)
        logging.info("added %s requests to waiting queue", len(prompts))
        # schedule from waiting queue to running queue
        finished_seqs = []
        while not self.scheduler.is_finished():
            # run step (call certain engine, e.x. NaiveEngine)
            step_finished_seqs = self._step()
            finished_seqs.extend(step_finished_seqs)
            self.scheduler.logging_queues()
        # decode the finished_seq
        return [self.tokenizer.decode(seq.token_ids[seq.input_len:], skip_special_tokens=True) for seq in finished_seqs]

    def _add_request(self, prompt: str):
        # apply tokenizer chat template
        prompt = {"role": "user", "content": prompt}
        prompt = self.tokenizer.apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        # convert str to token id via tokenizer
        token_ids = self.tokenizer.encode(prompt)
        # create a new Sequence
        seq = Sequence(token_ids)
        # call scheduler to add
        self.scheduler.add(seq)

    def _step(self) -> list[Sequence]:
        # schedule the running
        scheduled_task = self.scheduler.schedule()
        # run
        self.runner.run(scheduled_task)
        # post processing
        finished_seqs = self.scheduler.post_run(scheduled_task)
        return finished_seqs

