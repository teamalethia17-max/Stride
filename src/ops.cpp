#include "stride/tensor.hpp"
#include <stdexcept>

namespace Stride {

// Computes C = A * B (Simple Row-Major CPU Gemm)
// A: [M x K], B: [K x N], C: [M x N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.dtype() != DataType::F32 || B.dtype() != DataType::F32 || C.dtype() != DataType::F32) {
        throw std::runtime_error("Stride Ops Error: MatMul currently only supports F32 processing.");
    }

    auto shape_A = A.shape();
    auto shape_B = B.shape();
    auto shape_C = C.shape();

    if (shape_A.size() != 2 || shape_B.size() != 2 || shape_C.size() != 2) {
        throw std::runtime_error("Tensors must be 2D matrices.");
    }

    int64_t M = shape_A[0];
    int64_t K = shape_A[1];
    int64_t N = shape_B[1];

    if (shape_B[0] != K || shape_C[0] != M || shape_C[1] != N) {
        throw std::runtime_error("Matrix dimension mismatch for multiplication.");
    }

    const float* a_data = A.data_as<float>();
    const float* b_data = B.data_as<float>();
    float* c_data = C.data_as<float>();

    // Parallelize with OpenMP since your CMake config has it enabled
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            c_data[i * N + j] = sum;
        }
    }
}

} // namespace Stride
