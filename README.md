# Stride

# Stride

**A CPU-first, pure C++ LLM inference engine built from scratch.**

Stride is an open-source project exploring how to run large language models efficiently using native C++—with a focus on memory management, streaming model weights, and Mixture-of-Experts (MoE) architectures.

The long-term goal is to make local LLM inference more accessible on machines with limited RAM, without relying on heavyweight inference frameworks.

> **Project status:** Early development. Features are implemented incrementally; model compatibility, performance, and functionality must be verified against the current codebase.

---

## Vision

Modern language models can require enormous amounts of memory. Stride explores a different approach: keep memory usage controlled, load model data efficiently, and execute inference through a native C++ runtime.

Stride aims to become an educational and experimental foundation for understanding and building LLM inference systems from the ground up.

## Core Goals

- **Pure C++ runtime** — Build the inference engine in modern C++.
- **CPU-first inference** — Target systems without requiring a GPU.
- **Memory-aware execution** — Manage tensors, buffers, and model weights carefully.
- **Weight streaming** — Explore loading model weights from disk instead of keeping every expert resident in RAM.
- **MoE support** — Work toward routing tokens through selected experts in Mixture-of-Experts models.
- **Transformer inference** — Implement core operations such as attention, positional encoding, and feed-forward layers.
- **KV caching** — Reuse previously computed attention keys and values during token generation.
- **Transparent benchmarks** — Measure actual latency, throughput, and memory usage.
- **Open development** — Publish source code, tests, documentation, and verified results on GitHub.

## Planned Architecture

```text
                 STRIDE
                   |
          +--------+--------+
          |                 |
      Model Loader       Tokenizer
          |                 |
          +--------+--------+
                   |
             Inference Runtime
                   |
          +--------+--------+
          |                 |
     Transformer          MoE Router
       Layers                 |
          |              Expert Store
          |                 |
          +--------+--------+
                   |
             CPU Operations
                   |
             Text Generation
```

This diagram represents the intended architecture, not a claim that every component is already implemented.

## Technology Stack

| Component | Technology |
|---|---|
| Language | C++20 |
| Build system | CMake |
| Execution target | CPU |
| Testing | CTest |
| Version control | Git |
| Development | Cross-platform tooling where practical |

## Development Roadmap

### Phase 1 — Core Foundations
- [ ] CMake project and build system
- [ ] Tensor representation and memory management
- [ ] Basic tensor operations
- [ ] Numerical correctness tests

### Phase 2 — Transformer Runtime
- [ ] Linear layers
- [ ] Normalization
- [ ] Rotary positional embeddings (RoPE)
- [ ] Causal self-attention
- [ ] KV cache
- [ ] Feed-forward networks
- [ ] Token sampling

### Phase 3 — Model Support
- [ ] Model configuration parsing
- [ ] Tokenizer integration
- [ ] Safe weight-file parsing
- [ ] Support for explicitly documented open-weight model formats
- [ ] End-to-end inference tests

### Phase 4 — Memory and MoE
- [ ] Expert routing
- [ ] Disk-backed expert weights
- [ ] Memory-mapped loading where supported
- [ ] Bounded expert cache
- [ ] Cache statistics and memory tracking
- [ ] Tests for streaming and eviction behavior

### Phase 5 — Optimization
- [ ] Multithreaded CPU kernels
- [ ] Cache-aware matrix multiplication
- [ ] Quantized weight support
- [ ] Performance profiling
- [ ] Reproducible benchmarks

*Roadmap items are goals, not completed features.*

## Building

Stride uses CMake and a C++20-compatible compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

These commands apply once the corresponding CMake targets and tests are present in the repository.

## Intended CLI

The following commands describe the planned interface. They are usable only when implemented in the current build.

```bash
stride --help

stride inspect --model /path/to/model

stride run \
  --model /path/to/model \
  --prompt "Explain how a CPU executes a program." \
  --max-tokens 64

stride bench --model /path/to/model

stride cache-stats
```

## Model Compatibility

Stride is intended to support legally available open-weight language models.

Compatibility depends on the model architecture, tensor formats, tokenizer, weight format, and operations implemented in the current version. A model should not be considered supported until it has been loaded and tested successfully.

**Proprietary models:** Stride cannot run a proprietary model such as Claude using its name or API alone. Local execution requires compatible model weights and an implemented architecture.

Always review a model's license before using or redistributing its weights.

## Performance Philosophy

Stride prioritizes measured results over marketing claims.

Performance depends on:
- Model size and quantization
- CPU architecture and available instruction sets
- RAM capacity and memory bandwidth
- Storage speed and cache behavior
- Context length and KV-cache size
- Thread count and implementation quality

Streaming expert weights may reduce the amount of model data that must remain in RAM, but it does not eliminate computation or disk-I/O costs. Loading experts repeatedly can make generation significantly slower.

Benchmarks should report the hardware, model, quantization, context length, memory usage, and measured tokens per second.

## Project Principles

1. **Working code over impressive claims.**
2. **Correctness before optimization.**
3. **Measured performance over invented benchmarks.**
4. **Explicit model compatibility over vague promises.**
5. **Memory efficiency without hiding latency trade-offs.**
6. **Open development and reproducible testing.**

## Contributing

Stride is an evolving engineering project. Contributions involving correctness, testing, documentation, CPU optimization, model-format support, and memory management are welcome.

Before submitting a contribution:
1. Build the project.
2. Run the available tests.
3. Document new features and limitations.
4. Include benchmark details when reporting performance improvements.

## License

The project's license has not been specified yet. A license should be added before the repository is presented as open source or others are invited to reuse its code.

## Disclaimer

Stride is an independent software project. It is not affiliated with or endorsed by any proprietary AI model provider.

---

**Stride — Build the runtime. Understand the model. Own the inference.**
