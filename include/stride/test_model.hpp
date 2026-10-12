#pragma once

#include "stride/tensor.hpp"

namespace stride {

struct TinyModel {
    Tensor weights;
    Tensor bias;

    static TinyModel make();
    Tensor forward(const Tensor& x) const;
};

}  // namespace stride
