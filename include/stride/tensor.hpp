
#include "stride/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace stride {

std::int64_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::U8:  return 1;
    }

    throw std::invalid_argument("unsupported dtype");
}

std::string shape_str(const Shape& shape) {
    std::string result = "(";

    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) result += ", ";
        result += std::to_string(shape[i]);
    }

    if (shape.size() == 1) result += ",";
    result += ")";
    return result;
}

Result<std::int64_t> shape_numel_checked(const Shape& shape) {
    if (shape.empty()) {
        return Result<std::int64_t>::ok(1);
    }

    std::int64_t n = 1;

    for (const std::int64_t d : shape) {
        if (d <= 0) {
            return Result<std::int64_t>::error(
                "invalid dimension " + std::to_string(d) +
                " in shape " + shape_str(shape));
        }

        if (n > std::numeric_limits<std::int64_t>::max() / d) {
            return Result<std::int64_t>::error(
                "shape numel overflow for " + shape_str(shape));
        }

        n *= d;
    }

    return Result<std::int64_t>::ok(n);
}

Tensor::Tensor(Shape shape, DType dtype)
    : shape_(std::move(shape)),
      dtype_(dtype),
      storage_(std::make_shared<std::vector<float>>()) {
    const auto n = shape_numel_checked(shape_);

    if (!n) {
        throw StatusError(n.message());
    }

    numel_ = n.value();
    storage_->resize(static_cast<std::size_t>(numel_), 0.0f);
}

Tensor Tensor::from_f32(Shape shape, std::vector<float> values) {
    const auto n = shape_numel_checked(shape);

    if (!n) {
        throw StatusError(n.message());
    }

    if (static_cast<std::uint64_t>(n.value()) != values.size()) {
        throw std::invalid_argument(
            "value count does not match shape " + shape_str(shape));
    }

    Tensor result(std::move(shape), DType::F32);
    *result.storage_ = std::move(values);
    return result;
}

Tensor Tensor::zeros(Shape shape) {
    return Tensor(std::move(shape), DType::F32);
}

Tensor Tensor::ones(Shape shape) {
    Tensor result(std::move(shape), DType::F32);
    result.fill(1.0f);
    return result;
}

std::int64_t Tensor::dim(std::size_t i) const {
    if (i >= shape_.size()) {
        throw std::out_of_range("tensor dimension index out of range");
    }

    return shape_[i];
}

float* Tensor::data() {
    return storage_->data();
}

const float* Tensor::data() const {
    return storage_->data();
}

std::span<float> Tensor::span() {
    return *storage_;
}

std::span<const float> Tensor::span() const {
    return *storage_;
}

float& Tensor::at(std::int64_t i) {
    if (i < 0 || i >= numel_) {
        throw std::out_of_range("tensor index out of range");
    }

    return (*storage_)[static_cast<std::size_t>(i)];
}

const float& Tensor::at(std::int64_t i) const {
    if (i < 0 || i >= numel_) {
        throw std::out_of_range("tensor index out of range");
    }

    return (*storage_)[static_cast<std::size_t>(i)];
}

float& Tensor::at(std::int64_t r, std::int64_t c) {
    if (shape_.size() != 2) {
        throw std::invalid_argument("two-dimensional access requires a matrix");
    }

    if (r < 0 || r >= shape_[0] || c < 0 || c >= shape_[1]) {
        throw std::out_of_range("tensor matrix index out of range");
    }

    return at(r * shape_[1] + c);
}

const float& Tensor::at(std::int64_t r, std::int64_t c) const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("two-dimensional access requires a matrix");
    }

    if (r < 0 || r >= shape_[0] || c < 0 || c >= shape_[1]) {
        throw std::out_of_range("tensor matrix index out of range");
    }

    return at(r * shape_[1] + c);
}

Tensor Tensor::view(Shape new_shape) const {
    const auto n = shape_numel_checked(new_shape);

    if (!n) {
        throw StatusError(n.message());
    }

    if (n.value() != numel_) {
        throw std::invalid_argument(
            "cannot view shape " + shape_str(shape_) +
            " as " + shape_str(new_shape));
    }

    Tensor result;
    result.shape_ = std::move(new_shape);
    result.dtype_ = dtype_;
    result.numel_ = numel_;
    result.storage_ = storage_;
    return result;
}

Tensor Tensor::clone() const {
    Tensor result(shape_, dtype_);
    *result.storage_ = *storage_;
    return result;
}

Tensor& Tensor::fill(float value) {
    std::fill(storage_->begin(), storage_->end(), value);
    return *this;
}

Tensor& Tensor::zero() {
    return fill(0.0f);
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000u)) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1fu;
    std::uint32_t mantissa = h & 0x03ffu;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int e = -14;

            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }

            mantissa &= 0x03ffu;
            bits = sign |
                   (static_cast<std::uint32_t>(e + 127) << 23) |
                   (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent + (127 - 15);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t f32_to_f16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint16_t sign =
        static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 255) {
        if (mantissa == 0) return sign | 0x7c00u;

        return sign | 0x7c00u |
               static_cast<std::uint16_t>(mantissa >> 13) | 0x0200u;
    }

    const int half_exponent = static_cast<int>(exponent) - 127 + 15;

    if (half_exponent >= 31) {
        return sign | 0x7c00u;
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;

        mantissa |= 0x800000u;
        const int shift = 14 - half_exponent;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t remainder_mask =
            (std::uint32_t{1} << shift) - 1;
        const std::uint32_t remainder = mantissa & remainder_mask;
        const std::uint32_t halfway = std::uint32_t{1} << (shift - 1);

        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1u))) {
            ++half_mantissa;
        }

        return sign | static_cast<std::uint16_t>(half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;

    if (remainder > 0x1000u ||
        (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;

        if (half_mantissa == 0x400u) {
            half_mantissa = 0;

            if (half_exponent + 1 >= 31) {
                return sign | 0x7c00u;
            }

            return sign |
                   static_cast<std::uint16_t>((half_exponent + 1) << 10);
        }
    }

    return sign |
           static_cast<std::uint16_t>(half_exponent << 10) |
           static_cast<std::uint16_t>(half_mantissa);
}

}  // namespace stride
