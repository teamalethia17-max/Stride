#pragma once

#include "stride/tensor.hpp"

namespace stride::ops {

void require_same_shape(const Tensor& a, const Tensor& b, const char* where);
void require_ndim(const Tensor& t, std::size_t n, const char* where);

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& a, float s);
Tensor mul_scalar(const Tensor& a, float s);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matvec(const Tensor& matrix, const Tensor& vector);
Tensor transpose2d(const Tensor& x);
Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor softmax(const Tensor& x);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-5f);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
float max_elem(const Tensor& x);
float sum_elem(const Tensor& x);
Tensor sum_last_dim(const Tensor& x);
std::int64_t argmax(const Tensor& x);
Tensor cat_last(const Tensor& a, const Tensor& b);

}  // namespace stride::ops
