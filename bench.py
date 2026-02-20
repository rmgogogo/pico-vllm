import logging
import sys
from time import perf_counter

from transformers import AutoTokenizer, AutoModelForCausalLM

# from picovllm.naive.naive_engine import NaiveEngine
# from picovllm.naive.batch_engine import NaiveBatchEngine
# from picovllm.naive.dcache_engine import DynamicCacheEngine
# from picovllm.naive.scache_engine import StaticCacheEngine
# from picovllm.naive.pack_engine import PackEngine
# from picovllm.naive.managed_pack_engine import ManagedPackEngine
from picovllm.pico.engine import PicoEngine

def run(engine, prompts) -> float:
    ts = perf_counter()
    results = engine.generate(prompts)
    tspend = perf_counter() - ts
    for item in results:
        logging.info("generated:\n%s", item)
    return tspend

def main():
#    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model.eval()

    engines = [
        # NaiveEngine(model, tokenizer), 
        # NaiveBatchEngine(model, tokenizer), 
        # DynamicCacheEngine(model, tokenizer), 
        # StaticCacheEngine(model, tokenizer), 
        # PackEngine(model, tokenizer),
        # ManagedPackEngine(model, tokenizer),
        PicoEngine(model, tokenizer),
    ]
    
    prompts = [
        "1 labubu 2 = 3. 2 labubu 3 = 5. Reply in one sentence: what's 3 labubu 4?",
        "1 labubu 2 = 3. 2 labubu 3 = 5. Reply in one sentence: what's 5 labubu 6?",
        "Reply in one sentence: what's labubu?",
    ]

    time_spends = [
        run(engine, prompts)
        for engine in engines
    ]
    logging.info(time_spends)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout
    )
    main()
