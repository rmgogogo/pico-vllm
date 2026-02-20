from enum import Enum
from itertools import count

class SequenceStatus(Enum):
    # waiting -> prefilled -> decoding -> finished
    WAITING = 0
    PREFILLED = 1
    DECODING = 2
    FINISHED = 3

class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int]):
        self.id = next(Sequence.counter)
        self.token_ids = token_ids[:]
        self.input_len = len(token_ids)
        self.status = SequenceStatus.WAITING

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]
    
    def __getstate__(self):
        return (self.id, self.token_ids, self.input_len, self.status)
    
    def __setstate__(self, state):
        self.id, self.token_ids, self.input_len, self.status = state

    def __repr__(self):
        return f"Sequence(id={self.id}, tokens_len={len(self.token_ids)}, status={self.status})"

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
