import logging
from collections import deque
from dataclasses import dataclass

from picovllm.pico.sequence import Sequence, SequenceStatus

@dataclass
class ScheduledTask:
    seqs: list[Sequence]
    prefill: bool

class Scheduler:
    def __init__(self, config):
        self.config = config
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduledTask:
        scheduled_seqs = []

        # prefill scheduling
        num_prefill_tokens = 0
        while self.waiting:
            seq = self.waiting[0]
            if num_prefill_tokens + len(seq) > self.config.max_prefill_tokens:
                break
            num_prefill_tokens += len(seq)
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return ScheduledTask(seqs=scheduled_seqs, prefill=True)
        
        # decode scheduling
        num_decode_tokens = 0
        while self.running:
            seq = self.running[0]
            if num_decode_tokens + len(seq) > self.config.max_decode_tokens:
                break
            num_decode_tokens += len(seq)
            self.running.popleft()
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            # add it back to queue left, only remove from running queue after EOS
            self.running.extendleft(reversed(scheduled_seqs))
        return ScheduledTask(seqs=scheduled_seqs, prefill=False)
    
    def post_run(self, task: ScheduledTask) -> list[Sequence]:
        finished_seq = []
        for seq in task.seqs:
            if seq.status == SequenceStatus.FINISHED:
                self.running.remove(seq)
                finished_seq.append(seq)
        return finished_seq
    
    def logging_queues(self):
        logging.info("waiting: %s", self.waiting)
        logging.info("running: %s", self.running)
