#include "stride/tensor.hpp"
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

    Tensor t1({2, 3});
    t1.fill(5.0f);
    check(t1.numel() == 6, "numel");
    check_near(t1.data()[0], 5.0f, 1e-5f, "fill");

    Tensor t2 = Tensor::from_f32({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    check(t2.dim(0) == 2 && t2.dim(1) == 2, "shape");
    check_near(t2.at(1, 1), 4.0f, 1e-5f, "at");

    Tensor z = Tensor::zeros({3});
    check_near(z.data()[0], 0.0f, 1e-5f, "zeros");
    Tensor o = Tensor::ones({3});
    check_near(o.data()[1], 1.0f, 1e-5f, "ones");

    Tensor t3 = t2.clone();
    t3.data()[0] = 99.0f;
    check_near(t2.at(0, 0), 1.0f, 1e-5f, "clone_independent");

    std::cout << "All tensor tests passed\n";
    return 0;
}
