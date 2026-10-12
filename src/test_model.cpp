#include "stride/test_model.hpp"

#include <algorithm>
#include <cmath>

namespace stride {

TinyModel TinyModel::make() {
    TinyModel model;
    model.weights = Tensor::from_f32({3}, {0.5f, -0.25f, 1.0f});
    model.bias = Tensor::from_f32({1}, {0.25f});
    return model;
}

Tensor TinyModel::forward(const Tensor& x) const {
    if (x.ndim() != 1 || x.dim(0) != 3) {
        throw StatusError("TinyModel input must be rank-1 with 3 elements");
    }
    float acc = bias.data()[0];
    for (std::int64_t i = 0; i < x.dim(0); ++i) {
        acc += x.data()[i] * weights.data()[i];
    }
    Tensor out({1});
    out.data()[0] = std::max(0.0f, acc);
    return out;
}

}  // namespace stride
