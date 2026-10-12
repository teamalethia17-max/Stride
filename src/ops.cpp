#include "stride/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace stride::ops {

void require_same_shape(const Tensor& a, const Tensor& b, const char* where) {
    if (a.shape() != b.shape()) {
        throw StatusError(std::string(where) + ": shape mismatch " + shape_str(a.shape()) + " vs " + shape_str(b.shape()));
    }
}

void require_ndim(const Tensor& t, std::size_t n, const char* where) {
    if (t.ndim() != n) {
        throw StatusError(std::string(where) + ": expected rank " + std::to_string(n) + ", got " + shape_str(t.shape()));
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "add");
    Tensor c(a.shape());
    for (std::int64_t i = 0; i < a.numel(); ++i) c.data()[i] = a.data()[i] + b.data()[i];
    return c;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "mul");
    Tensor c(a.shape());
    for (std::int64_t i = 0; i < a.numel(); ++i) c.data()[i] = a.data()[i] * b.data()[i];
    return c;
}

Tensor add_scalar(const Tensor& a, float s) {
    Tensor c(a.shape());
    for (std::int64_t i = 0; i < a.numel(); ++i) c.data()[i] = a.data()[i] + s;
    return c;
}

Tensor mul_scalar(const Tensor& a, float s) {
    Tensor c(a.shape());
    for (std::int64_t i = 0; i < a.numel(); ++i) c.data()[i] = a.data()[i] * s;
    return c;
}

static Tensor gemm_nn(const float* a, const float* b, std::int64_t m, std::int64_t k, std::int64_t n) {
    Tensor c({m, n});
    float* out = c.data();
    std::fill(out, out + m * n, 0.0f);
#ifdef STRIDE_HAS_OPENMP
#pragma omp parallel for
#endif
    for (std::int64_t i = 0; i < m; ++i) {
        const float* ai = a + i * k;
        float* ci = out + i * n;
        for (std::int64_t p = 0; p < k; ++p) {
            const float aik = ai[p];
            const float* bp = b + p * n;
            for (std::int64_t j = 0; j < n; ++j) ci[j] += aik * bp[j];
        }
    }
    return c;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() == 2 && b.ndim() == 2) {
        if (a.dim(1) != b.dim(0)) throw StatusError("matmul inner dim mismatch");
        return gemm_nn(a.data(), b.data(), a.dim(0), a.dim(1), b.dim(1));
    }
    if (a.ndim() == 3 && (b.ndim() == 2 || b.ndim() == 3)) {
        const std::int64_t batch = a.dim(0);
        const std::int64_t m = a.dim(1), k = a.dim(2);
        const bool b_batched = b.ndim() == 3;
        const std::int64_t k2 = b_batched ? b.dim(1) : b.dim(0);
        const std::int64_t n = b_batched ? b.dim(2) : b.dim(1);
        if (k != k2) throw StatusError("batched matmul inner dim mismatch");
        Tensor c({batch, m, n});
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            const float* ap = a.data() + bi * m * k;
            const float* bp = b_batched ? b.data() + bi * k * n : b.data();
            Tensor slice = gemm_nn(ap, bp, m, k, n);
            std::copy(slice.data(), slice.data() + m * n, c.data() + bi * m * n);
        }
        return c;
    }
    throw StatusError("matmul unsupported ranks");
}

Tensor matvec(const Tensor& matrix, const Tensor& vector) {
    require_ndim(matrix, 2, "matvec");
    require_ndim(vector, 1, "matvec");
    if (matrix.dim(1) != vector.numel()) throw StatusError("matvec dim mismatch");
    Tensor y({matrix.dim(0)});
    for (std::int64_t i = 0; i < matrix.dim(0); ++i) {
        float acc = 0.0f;
        const float* row = matrix.data() + i * matrix.dim(1);
        for (std::int64_t k = 0; k < matrix.dim(1); ++k) acc += row[k] * vector.data()[k];
        y.data()[i] = acc;
    }
    return y;
}

Tensor transpose2d(const Tensor& x) {
    require_ndim(x, 2, "transpose2d");
    Tensor out({x.dim(1), x.dim(0)});
    for (std::int64_t i = 0; i < x.dim(0); ++i)
        for (std::int64_t j = 0; j < x.dim(1); ++j)
            out.at(j, i) = x.at(i, j);
    return out;
}

Tensor silu(const Tensor& x) {
    Tensor out(x.shape());
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        out.data()[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

Tensor gelu(const Tensor& x) {
    constexpr float k = 0.7978845608f;
    Tensor out(x.shape());
    for (std::int64_t i = 0; i < x.numel(); ++i) {
        const float v = x.data()[i];
        out.data()[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    if (x.ndim() == 0) throw StatusError("softmax on scalar");
    const std::int64_t d = x.dim(x.ndim() - 1);
    const std::int64_t rows = x.numel() / d;
    Tensor out(x.shape());
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = x.data() + r * d;
        float* o = out.data() + r * d;
        float mx = *std::max_element(row, row + d);
        float sum = 0.0f;
        for (std::int64_t i = 0; i < d; ++i) { o[i] = std::exp(row[i] - mx); sum += o[i]; }
        const float inv = 1.0f / sum;
        for (std::int64_t i = 0; i < d; ++i) o[i] *= inv;
    }
    return out;
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    const std::int64_t d = x.dim(x.ndim() - 1);
    if (weight.numel() != d) throw StatusError("rms_norm weight size mismatch");
    const std::int64_t rows = x.numel() / d;
    Tensor out(x.shape());
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = x.data() + r * d;
        float ms = 0.0f;
        for (std::int64_t i = 0; i < d; ++i) ms += row[i] * row[i];
        const float scale = 1.0f / std::sqrt(ms / static_cast<float>(d) + eps);
        float* o = out.data() + r * d;
        for (std::int64_t i = 0; i < d; ++i) o[i] = row[i] * scale * weight.data()[i];
    }
    return out;
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    const std::int64_t d = x.dim(x.ndim() - 1);
    if (weight.numel() != d || bias.numel() != d) throw StatusError("layer_norm size mismatch");
    const std::int64_t rows = x.numel() / d;
    Tensor out(x.shape());
    for (std::int64_t r = 0; r < rows; ++r) {
        const float* row = x.data() + r * d;
        float mean = 0.0f;
        for (std::int64_t i = 0; i < d; ++i) mean += row[i];
        mean /= static_cast<float>(d);
        float var = 0.0f;
        for (std::int64_t i = 0; i < d; ++i) { const float v = row[i] - mean; var += v * v; }
        const float scale = 1.0f / std::sqrt(var / static_cast<float>(d) + eps);
        float* o = out.data() + r * d;
        for (std::int64_t i = 0; i < d; ++i) o[i] = (row[i] - mean) * scale * weight.data()[i] + bias.data()[i];
    }
    return out;
}

float max_elem(const Tensor& x) {
    if (x.empty()) throw StatusError("max_elem empty");
    return *std::max_element(x.data(), x.data() + x.numel());
}

float sum_elem(const Tensor& x) {
    return std::accumulate(x.data(), x.data() + x.numel(), 0.0f);
}

Tensor sum_last_dim(const Tensor& x) {
    const std::int64_t d = x.dim(x.ndim() - 1);
    const std::int64_t rows = x.numel() / d;
    Shape out_shape = x.shape();
    out_shape.back() = 1;
    Tensor out(out_shape);
    for (std::int64_t r = 0; r < rows; ++r) {
        float s = 0.0f;
        for (std::int64_t i = 0; i < d; ++i) s += x.data()[r * d + i];
        out.data()[r] = s;
    }
    return out;
}

std::int64_t argmax(const Tensor& x) {
    if (x.empty()) throw StatusError("argmax empty");
    return static_cast<std::int64_t>(std::max_element(x.data(), x.data() + x.numel()) - x.data());
}

Tensor cat_last(const Tensor& a, const Tensor& b) {
    if (a.ndim() != b.ndim()) throw StatusError("cat_last rank mismatch");
    for (std::size_t i = 0; i + 1 < a.ndim(); ++i) {
        if (a.dim(i) != b.dim(i)) throw StatusError("cat_last prefix mismatch");
    }
    const std::int64_t da = a.dim(a.ndim() - 1);
    const std::int64_t db = b.dim(b.ndim() - 1);
    Shape out_shape = a.shape();
    out_shape.back() = da + db;
    Tensor out(out_shape);
    const std::int64_t rows = a.numel() / da;
    const std::int64_t dc = da + db;
    for (std::int64_t r = 0; r < rows; ++r) {
        std::copy(a.data() + r * da, a.data() + r * da + da, out.data() + r * dc);
        std::copy(b.data() + r * db, b.data() + r * db + db, out.data() + r * dc + da);
    }
    return out;
}

}  // namespace stride::ops
