# pico-vllm

**A minimalist, educational deep-dive into the vLLM architecture.**

`pico-vllm` is a streamlined implementation of the vLLM inference engine, designed specifically for researchers and engineers who want to understand the "magic" behind high-throughput LLM serving without getting lost in low-level hardware optimizations or complex C++ abstractions.

### 💡 Philosophy: Human Readability First
While most production inference engines prioritize hardware-specific micro-optimizations, **pico-vllm prioritizes human understanding.** This repository provides a clean, Pythonic reference implementation that is easy to read, modify, and extend.

---

## 🚀 Key Features
- **Continuous Batching:** Iteration-level scheduling to maximize throughput.
- **KV Caching:** Efficient memory management for incremental token decoding.
- **Prefill & Decode:** Optimized orchestration of prompt processing and generation phases.
- **Zero-CUDA Dependency:** Designed for rapid research and experimentation on any environment.

## 🗺️ Future Roadmap
- [ ] **Advanced Attention:** PagedAttention and FlashAttention (MLX backend).
- [ ] **Graph Capture:** Performance gains via execution graph optimization.
- [ ] **Distributed Inference:** Tensor, Pipeline, and Model Parallelism.
- [ ] **Production Layer:** Server-side rate limiting and OpenAI-compatible API protocol.

## 🛠️ Getting Started

This project uses `uv` for efficient environment and dependency management. Refer to the `Makefile` for standard operations.

```bash
# Setup the environment and sync dependencies
make sync

# Run the benchmark sandbox
make run
```

## 📚 References

- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM TPU](https://github.com/vllm-project/tpu-inference)
- [vLLM MLX](https://github.com/waybarrios/vllm-mlx)
- [vLLM Nano](https://github.com/GeeeekExplorer/nano-vllm)
- [Nano AIGC](https://github.com/rmgogogo/nano-aigc)
- [SGLang](https://github.com/sgl-project/sglang)
- [MLX](https://github.com/ml-explore/mlx)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)


❤️⾺❤️ Made with love this Lunar New Year.