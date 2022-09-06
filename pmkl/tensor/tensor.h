#pragma once

#include <iostream>
#include <vector>

#include "exception.h"
#include "array.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "device.h"
#include "launcher.h"

namespace pmkl {

namespace memory {

void delete_impl(void *ctx) {
    GpuLauncher::GetInstance()->free<char>((char *)ctx);
}

class TensorStorage : public intrusive_ptr_target {
protected:
    DataPtr ptr_;
    size_t size_;

public:
    TensorStorage(Device d, size_t size) :
        size_(size) {
        auto raw_ptr = (void *)GpuLauncher::GetInstance()->malloc<char>(size_);
        DataPtr ptr(raw_ptr, raw_ptr, delete_impl, d);
        ptr_ = std::move(ptr);
    }
    ~TensorStorage() {
        ptr_.clear();
    }
};

} // namespace memory

#define MAX_TENSOR_DIM 12

class Tensor;
Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, Device d);

class Tensor {
    using dim_array_t = memory::array<uint64_t, MAX_TENSOR_DIM>;
    int dim_;
    dim_array_t shape_;
    dim_array_t stride_;
    ScalarType dtype_;
    Device device_;

    uint64_t numel_;
    intrusive_ptr<memory::TensorStorage> storage_;

    void new_storage_() {
        size_t bytes = shape_[0] * stride_[0] * element_size(dtype_);
        storage_.set_ptr(new memory::TensorStorage(device_, bytes));
    }
    friend Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, Device d);

public:
    Tensor(std::vector<uint64_t> &shape, ScalarType dtype, Device d) :
        dtype_(dtype), device_(d) {
        CHECK_FAIL(shape.size() <= MAX_TENSOR_DIM);
        dim_ = shape.size();
        numel_ = 1;
        for (int i = 0; i < dim_; i++) {
            stride_[dim_ - 1 - i] = numel_;
            numel_ *= shape[i];
            shape_[i] = shape[i];
        }
    }

    Tensor(const Tensor &other) noexcept :
        dim_(other.dim_), shape_(other.shape_), stride_(other.stride_),
        dtype_(other.dtype_), device_(other.device_), numel_(other.numel_),
        storage_(other.storage_) {
    }

    ~Tensor() {
    }

    uint64_t numel() const {
        return numel_;
    }
    int dim() const {
        return dim_;
    }
    uint64_t size(int d) const {
        return shape_[d];
    }
};

Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, Device d) {
    Tensor output(shape, dtype, d);
    output.new_storage_();
    return output;
}

} // namespace pmkl