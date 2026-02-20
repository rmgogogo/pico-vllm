# Project Overview

This repository is designed for studying and benchmarking various LLM inference strategies without CUDA dependencies.

## Architecture and Codebase

- `picovllm/naive/`: Contains sandbox code for rapid research, experimentation, and API familiarization.
- `picovllm/pico/`: The core implementation, structured as follows:
  - **Engine**: The primary module and entry point for the system.
  - **Scheduler**: Manages dual queues and orchestrates workload scheduling.
  - **Runner**: Processes tasks scheduled by the scheduler.
  - **Sequence**: The primary data structure, representing a list of token IDs.

## Project Mandates

### DO

- **Use the Makefile**: Always refer to the `Makefile` for running the application; it utilizes `uv` for environment and dependency management.

### DO NOT

- **No Test Cases**: Do not author or commit any test cases within this project.
