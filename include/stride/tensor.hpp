#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <span>
#include <variant>

namespace stride {

enum class DType : std::uint8_t {
    F32 = 0,
    F16 = 1,
    U8 = 2,
};

using Shape = std::vector<std::int64_t>;

std::int64_t dtype_size(DType dtype);
std::string shape_str(const Shape& shape);

class StatusError : public std::exception {
public:
    explicit StatusError(const std::string& msg) : msg_(msg) {}
    const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

class Status {
public:
    static Status ok() { return Status(true, {}); }
    static Status error(const std::string& msg) { return Status(false, msg); }
    
    explicit operator bool() const { return ok_; }
    std::string message() const { return msg_; }
private:
    Status(bool ok, const std::string& msg) : ok_(ok), msg_(msg) {}
    bool ok_;
    std::string msg_;
};

template <typename T>
class Result {
public:
    static Result ok(T value) { return Result(true, value, {}); }
    static Result error(const std::string& msg) { return Result(false, T(), msg); }
    
    explicit operator bool() const { return ok_; }
    const T& value() const { return value_; }
    T take() { return std::move(value_); }
    Status status() const { return ok_ ? Status::ok() : Status::error(msg_); }
    std::string message() const { return msg_; }
private:
    Result(bool ok, T value, const std::string& msg) 
        : ok_(ok), value_(std::move(value)), msg_(msg) {}
    bool ok_;
    T value_;
    std::string msg_;
};

class Tensor {
public:
    explicit Tensor(Shape shape = {}, DType dtype = DType::F32);
    
    static Tensor from_f32(Shape shape, std::vector<float> values);
    static Tensor zeros(Shape shape);
    static Tensor ones(Shape shape);
    
    const Shape& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    std::int64_t numel() const { return numel_; }
    std::size_t ndim() const { return shape_.size(); }
    bool empty() const { return numel_ == 0; }
    std::int64_t dim(std::size_t i) const;
    std::int64_t nbytes() const { return numel_ * static_cast<std::int64_t>(sizeof(float)); }
    
    float* data();
    const float* data() const;
    
    std::span<float> span();
    std::span<const float> span() const;
    
    float& at(std::int64_t i);
    const float& at(std::int64_t i) const;
    float& at(std::int64_t r, std::int64_t c);
    const float& at(std::int64_t r, std::int64_t c) const;
    
    Tensor view(Shape new_shape) const;
    Tensor clone() const;
    
    Tensor& fill(float value);
    Tensor& zero();

private:
    Shape shape_;
    DType dtype_;
    std::int64_t numel_;
    std::shared_ptr<std::vector<float>> storage_;
};

Result<std::int64_t> shape_numel_checked(const Shape& shape);
float f16_to_f32(std::uint16_t h);
std::uint16_t f32_to_f16(float value);

}  // namespace stride
