#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace Stride {

enum class DataType {
    F32,
    F16,
    I32
};

// Global helpers: DECLARE only in header (or mark inline if defined here)
size_t dtype_size(DataType dtype);
std::string shape_str(const std::vector<int64_t>& shape);
float f16_to_f32(uint16_t h);
uint16_t f32_to_f16(float f);

class Tensor {
public:
    // Constructors
    Tensor(std::vector<int64_t> shape, DataType dtype);
    ~Tensor() = default;

    // Rule of 5 (Disable or implement properly to avoid memory ownership issues)
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Observers
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int64_t numel() const { return numel_; }
    size_t size_bytes() const { return numel_ * dtype_size(dtype_); }
    
    // Raw Data Access
    void* data() { return storage_.get(); }
    const void* data() const { return storage_.get(); }

    template<typename T>
    T* data_as() { return reinterpret_cast<T*>(storage_.get()); }

    template<typename T>
    const T* data_as() const { return reinterpret_cast<const T*>(storage_.get()); }

private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    int64_t numel_;
    std::unique_ptr<uint8_t[]> storage_; // Heap memory allocation container
};

} // namespace Stride
