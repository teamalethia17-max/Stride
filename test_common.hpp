#pragma once
#include <cmath>
#include <cstdlib>
#include <iostream>

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " #cond "\n"; \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                 \
    do {                                                                                      \
        const float _va = static_cast<float>(a);                                              \
        const float _vb = static_cast<float>(b);                                              \
        if (std::fabs(_va - _vb) > (eps)) {                                                   \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " |" << _va << "-" << _vb \
                      << "| > " << (eps) << "\n";                                             \
            std::abort();                                                                     \
        }                                                                                     \
    } while (0)
