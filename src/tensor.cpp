#include "stride/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace stride {

std::string shape_str(const Shape& shape) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) oss << ", ";
        oss << shape[i];
    }
    oss << ']';
    return oss.str();
}

std::int64_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::U8: return 1;
    }
    return 4;
}

Result<std::int64_t> shape_numel_checked(const Shape& shape) {
    if (shape.empty()) return Result<std::int64_t>::ok(std::int64_t{1});
    std::int64_t n = 1;
    for (std::int64_t d : shape) {
        if (d <= 0) return Result<std::int64_t>::error("invalid dimension " + std::to_string(d) + " in shape " + shape_str(shape));
        if (n > (std::numeric_limits<std::int64_t>::max() / d)) return Result<std::int64_t>::error("shape numel overflow for " + shape_str(shape));
        n *= d;
    }
    return Result<std::int64_t>::ok(n);
}

float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) & 1u;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x3ffu;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::uint16_t f32_to_f16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant = (mant | 0x800000u) >> (1 - exp);
        return static_cast<std::uint16_t>(sign | (mant >> 13));
    }
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

Tensor::Tensor(Shape shape, DType dtype) : shape_(std::move(shape)), dtype_(dtype) {
    if (dtype != DType::F32) throw StatusError("Tensor compute storage is f32; convert other dtypes on load");
    auto n = shape_numel_checked(shape_);
    if (!n) throw StatusError(n.message());
    numel_ = n.value();
    storage_ = std::make_shared<std::vector<float>>(static_cast<std::size_t>(numel_), 0.0f);
}

Tensor Tensor::from_f32(Shape shape, std::vector<float> values) {
    auto n = shape_numel_checked(shape);
    if (!n) throw StatusError(n.message());
    if (static_cast<std::int64_t>(values.size()) != n.value()) {
        throw StatusError("from_f32: shape " + shape_str(shape) + " needs " + std::to_string(n.value()) + " values, got " + std::to_string(values.size()));
    }
    Tensor t;
    t.shape_ = std::move(shape);
    t.dtype_ = DType::F32;
    t.numel_ = n.value();
    t.storage_ = std::make_shared<std::vector<float>>(std::move(values));
    return t;
}

Tensor Tensor::zeros(Shape shape) { return Tensor(std::move(shape)); }

Tensor Tensor::ones(Shape shape) {
    Tensor t(std::move(shape));
    t.fill(1.0f);
    return t;
}

std::int64_t Tensor::dim(std::size_t i) const {
    if (i >= shape_.size()) throw StatusError("dim " + std::to_string(i) + " out of range for " + shape_str(shape_));
    return shape_[i];
}

float* Tensor::data() {
    if (!storage_) throw StatusError("Tensor::data on empty tensor");
    return storage_->data();
}

const float* Tensor::data() const {
    if (!storage_) throw StatusError("Tensor::data on empty tensor");
    return storage_->data();
}

std::span<float> Tensor::span() { return {data(), static_cast<std::size_t>(numel_)}; }
std::span<const float> Tensor::span() const { return {data(), static_cast<std::size_t>(numel_)}; }

float& Tensor::at(std::int64_t i) {
    if (i < 0 || i >= numel_) throw StatusError("index out of range");
    return data()[i];
}

const float& Tensor::at(std::int64_t i) const {
    if (i < 0 || i >= numel_) throw StatusError("index out of range");
    return data()[i];
}

float& Tensor::at(std::int64_t r, std::int64_t c) {
    if (ndim() != 2) throw StatusError("2D at() requires rank-2 tensor");
    return at(r * shape_[1] + c);
}

const float& Tensor::at(std::int64_t r, std::int64_t c) const {
    if (ndim() != 2) throw StatusError("2D at() requires rank-2 tensor");
    return at(r * shape_[1] + c);
}

Tensor Tensor::view(Shape new_shape) const {
    auto n = shape_numel_checked(new_shape);
    if (!n) throw StatusError(n.message());
    if (n.value() != numel_) throw StatusError("view size mismatch: " + shape_str(shape_) + " -> " + shape_str(new_shape));
    Tensor t;
    t.shape_ = std::move(new_shape);
    t.dtype_ = dtype_;
    t.numel_ = numel_;
    t.storage_ = storage_;
    return t;
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_);
    if (numel_ > 0) std::copy(data(), data() + numel_, t.data());
    return t;
}

Tensor& Tensor::fill(float value) {
    if (numel_ > 0) std::fill(data(), data() + numel_, value);
    return *this;
}

Tensor& Tensor::zero() { return fill(0.0f); }

}  // namespace stride
