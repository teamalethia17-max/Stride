# Stride v1.0.0

**A pure C++20 CPU-first LLM inference engine for minimal deployments.**

Stride is a minimal, verifiable implementation of inference kernels and deterministic execution for small transformer models on CPU. Emphasis is on correctness, clarity, and reproducibility rather than performance.

**Project status:** v1.0.0 release. Core components build, test, and run. Pretrained model support is not yet implemented. See [Limitations](#limitations) and [Future Work](#future-work).

---

## What Works in v1.0.0

- **C++20 build system** — CMake 3.20+, builds cleanly on GCC, Clang, MSVC.
- **Tensor abstraction** — Shaped float32 storage with bounds checking and clear error messages.
- **Tensor operations** — add, mul, matmul, softmax, silu, gelu, rms_norm, layer_norm, transpose, argmax, and more.
- **Deterministic test model** — A tiny neural network (3 inputs, 1 output, 4 weights + bias) that runs real computation with verifiable results.
- **CLI interface** — `stride --help`, `stride --version`, `stride test-model`.
- **Automated tests** — Unit tests for tensor and ops. CLI sanity tests. All runnable via CTest.
- **GitHub Actions** — Reproducible builds on Linux, macOS, Windows.

## What Does NOT Work Yet

- **Model file loading** (GGUF, safetensors, etc.) — Code structure exists but no weight loading is implemented.
- **Real pretrained models** — No inference on actual LLMs. The test model is 4 parameters; real models are millions to billions.
- **Tokenizers** — No text-to-token or token-to-text conversion.
- **Text generation** — No autoregressive sampling or prompt processing.
- **KV caching, attention, embeddings** — Core transformer layers exist as headers but are not wired into a working model pipeline.
- **Quantization, MoE, weight streaming** — Future phases.

---

## Quick Start

### Prerequisites

- CMake 3.20 or later
- C++20 compiler (GCC 10+, Clang 13+, MSVC 2019+)
- Standard build tools (make, ninja, or VS)
- Optional: OpenMP for parallelization

### Build

```bash
git clone https://github.com/teamalethia17-max/Stride.git
cd Stride
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run Tests

```bash
ctest --test-dir build --output-on-failure
```

Expected output:
```
Test project build
    Start 1: test_tensor
1/4 Test #1: test_tensor ....................... Passed    0.01 sec
    Start 2: test_ops
2/4 Test #2: test_ops .......................... Passed    0.01 sec
    Start 3: cli_help
3/4 Test #3: cli_help .......................... Passed    0.00 sec
    Start 4: cli_version
4/4 Test #4: cli_version ....................... Passed    0.00 sec
    Start 5: cli_test_model
5/5 Test #5: cli_test_model .................... Passed    0.00 sec

100% tests passed, 0 tests failed out of 5
```

### Run CLI

```bash
./build/stride --help
./build/stride --version
./build/stride test-model
```

Example output from `test-model`:
```
test-model: input [1, 2, 3]
test-model: output [2.25]
test-model: OK
```

---

## Architecture

```
Stride v1.0.0
    |
    +-- Tensor (shape, dtype, float32 storage)
    |       |
    |       +-- from_f32(), zeros(), ones()
    |       +-- shape(), numel(), data()
    |       +-- at(i), at(r, c)
    |       +-- view(), clone(), fill()
    |
    +-- Ops (element-wise and linear algebra)
    |       |
    |       +-- add, mul, matmul, matvec
    |       +-- softmax, silu, gelu, argmax
    |       +-- rms_norm, layer_norm
    |
    +-- TinyModel (deterministic 3→1 linear + ReLU)
    |       |
    |       +-- make() — create with fixed weights
    |       +-- forward(x) — compute output
    |
    +-- CLI (stride executable)
            |
            +-- --help, --version
            +-- test-model (runs TinyModel)
```

Layers not yet integrated into a pipeline:
- MultiHeadAttention (headers only)
- RotaryEmbeddings (headers only)
- KVCache (headers only)
- Linear layers (headers only)
- Embeddings (headers only)
- Tokenizer (headers only)

---

## Project Structure

```
Stride/
├── CMakeLists.txt           # Main build configuration
├── CMakePresets.json        # Build presets for IDE integration
├── README.md                # This file
├── include/stride/
│   ├── tensor.hpp           # Tensor class, StatusError, Result<T>
│   ├── ops.hpp              # Operation declarations
│   ├── dtype.hpp            # DType enum (F32, F16, U8)
│   └── test_model.hpp       # Tiny deterministic model
├── src/
│   ├── main.cpp             # CLI entry point
│   ├── tensor.cpp           # Tensor implementation
│   ├── ops.cpp              # Operations implementation
│   └── test_model.cpp       # TinyModel implementation
├── tests/
│   ├── test_tensor.cpp      # Unit tests for Tensor
│   └── test_ops.cpp         # Unit tests for Ops
├── .github/workflows/
│   └── c-cpp.yml            # GitHub Actions CI/CD
└── LICENSE                  # MIT License
```

---

## Testing Strategy

Stride v1.0.0 includes two types of tests:

1. **Unit tests** (test_tensor, test_ops)
   - Verify tensor creation, access, and modification.
   - Verify operation correctness (add, matmul, softmax, etc.).
   - All tests use deterministic inputs and check exact (or near) expected outputs.

2. **Integration tests** (CLI tests)
   - `cli_help` — Ensure help text is printed.
   - `cli_version` — Ensure version is printed.
   - `cli_test_model` — Run the tiny model on a fixed input and verify output.

All tests are run via CTest and must pass before release.

---

## Implementation Notes

### Tensor Storage

Tensors store only float32 data. The `DType` enum supports F32, F16, and U8 at the metadata level, but all computations use float32. External weight formats (GGUF, etc.) will be converted to float32 on load.

### Error Handling

All errors throw `stride::StatusError` with descriptive messages. No silent failures or undefined behavior.

### Bounds Checking

All tensor accesses check bounds and throw on out-of-range index. All shape operations check for overflow.

### Deterministic Computation

The tiny model uses fixed, hardcoded weights. Its output is deterministic and verifiable: `TinyModel.forward([1, 2, 3])` always produces `[2.25]`.

---

## Limitations

**Stride v1.0.0 is a minimal proof-of-concept, not a general inference engine.** Known limitations:

1. **No external model loading** — Cannot load GGUF, safetensors, or other model files yet.
2. **No tokenization** — Cannot process text input.
3. **No text generation** — No sampling, no autoregressive decoding.
4. **Tiny test model only** — Cannot run real LLMs.
5. **CPU-only** — No GPU support.
6. **Float32 compute only** — No quantization or mixed precision.
7. **No streaming or disk-backed weights** — All tensors must fit in RAM.
8. **Single-threaded by default** — OpenMP support is optional; multi-threading is not automatic.
9. **Minimal performance tuning** — Focuses on correctness, not speed. Use established frameworks (llama.cpp, vLLM) for production.

---

## Future Work

Planned phases (not part of v1.0.0):

### Phase 2: Model Loading
- Implement GGUF tensor reading (metadata, dimensions, offset validation).
- Parse model config from GGUF metadata.
- Load weights into Tensor objects.

### Phase 3: Transformer Inference
- Integrate MultiHeadAttention, RoPE, LayerNorm.
- Implement a single decoder block.
- Add a simple Llama-like forward pass.
- Token sampling (temperature, top-k, top-p).

### Phase 4: Tokenization
- Implement BPE tokenizer or integrate a compatible one.
- Add text encoding/decoding.

### Phase 5: Full Inference Loop
- Prompt processing.
- Autoregressive text generation.
- KV caching for efficiency.
- Support for a documented, open-weight model (e.g., TinyLlama, Llama 2 7B).

### Phase 6+: Optimization, MoE, Streaming
- Performance optimization and benchmarking.
- Mixture-of-Experts routing.
- Disk-backed weight streaming.
- Advanced quantization support.

---

## Contributing

This is an educational and experimental project. Contributions are welcome. Please ensure:

1. Code builds cleanly with C++20 and CMake 3.20+.
2. All tests pass (CTest).
3. Changes preserve or improve existing correctness.
4. New features include tests.
5. Documentation is updated.

---

## License

MIT License. See LICENSE file for details.

---

## Contact

For issues, questions, or contributions, open an issue or pull request on GitHub.

**Stride 1.0.0** — October 12, 2026.
