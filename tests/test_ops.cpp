#include "stride/tensor.hpp"
#include "stride/ops.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << "\n";
        std::exit(1);
    }
}

void check_near(float a, float b, float eps, const char* name) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "FAIL: " << name << " (" << a << " != " << b << ")\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    using stride::Tensor;
    using stride::ops::add;
    using stride::ops::argmax;
    using stride::ops::matmul;
    using stride::ops::mul;
    using stride::ops::softmax;

    Tensor a = Tensor::from_f32({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_f32({2, 2}, {1, 1, 1, 1});
    Tensor c = add(a, b);
    check_near(c.at(1, 1), 5.0f, 1e-5f, "add");

    Tensor d = mul(a, b);
    check_near(d.at(1, 1), 4.0f, 1e-5f, "mul");

    Tensor m = Tensor::from_f32({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor n = Tensor::from_f32({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor mm = matmul(m, n);
    check(mm.dim(0) == 2 && mm.dim(1) == 2, "matmul_shape");
    check_near(mm.at(0, 0), 58.0f, 1e-5f, "matmul");

    Tensor logits = Tensor::from_f32({3}, {0.0f, 0.0f, 0.0f});
    Tensor probs = softmax(logits);
    check_near(probs.data()[0] + probs.data()[1] + probs.data()[2], 1.0f, 1e-4f, "softmax_sum");
    check_near(probs.data()[0], 1.0f / 3.0f, 1e-4f, "softmax_uniform");

    Tensor maxv = Tensor::from_f32({4}, {1, 5, 2, 3});
    check(argmax(maxv) == 1, "argmax");

    std::cout << "All ops tests passed\n";
    return 0;
}
