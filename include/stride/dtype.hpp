#pragma once

#include <cstdint>
#include <string>

namespace stride {

enum class DType : std::uint8_t {
    F32 = 0,
    F16 = 1,
    U8 = 2,
};

inline std::int64_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::U8: return 1;
    }
    return 4;  // default to F32
}

}  // namespace stride
