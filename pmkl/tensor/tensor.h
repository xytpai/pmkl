#pragma once

#include <iostream>
#include <vector>

#include "exception.h"
#include "array.h"
#include "data_ptr.h"
#include "intrusive_ptr.h"
#include "scalar_type.h"
#include "launcher.h"

namespace pmkl {

namespace memory {

void delete_impl(void *ctx) {
    GpuLauncher::GetInstance()->free(ctx);
}

class TensorStorage : public intrusive_ptr_target {
protected:
    DataPtr ptr_;
    size_t size_;
    int device_;

public:
    TensorStorage() :
        ptr_(), size_(0), device_(0) {
    }
    TensorStorage(size_t size, int device) :
        size_(size), device_(device) {
        auto l = GpuLauncher::GetInstance();
        if (l->device() != device_) l->set_device(device_);
        auto raw_ptr = (void *)l->malloc<char>(size_);
        DataPtr ptr(raw_ptr, raw_ptr, delete_impl);
        ptr_ = std::move(ptr);
    }
    ~TensorStorage() {
        auto l = GpuLauncher::GetInstance();
        if (l->device() != device_) l->set_device(device_);
        ptr_.clear();
    }
    int device() const {
        return device_;
    }
    void *data_ptr() const {
        return ptr_.get();
    }
    bool defined() const {
        return static_cast<bool>(ptr_);
    }
    size_t size() const {
        return size_;
    }
};

} // namespace memory

#define MAX_TENSOR_DIM 12

using namespace memory;

class Tensor;
Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, int device = 0);
Tensor zeros(std::vector<uint64_t> shape, ScalarType dtype, int device = 0);

class Tensor {
    using dim_array_t = memory::array<uint64_t, MAX_TENSOR_DIM>;
    int dim_;
    dim_array_t shape_;
    dim_array_t stride_;
    ScalarType dtype_;
    uint64_t numel_;

    intrusive_ptr<memory::TensorStorage> storage_;

    void new_storage_(int device) {
        size_t bytes = shape_[0] * stride_[0] * element_size(dtype_);
        storage_.unsafe_set_ptr(new memory::TensorStorage(bytes, device));
    }
    friend Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, int device);
    friend Tensor zeros(std::vector<uint64_t> shape, ScalarType dtype, int device);

public:
    Tensor(std::vector<uint64_t> &shape, ScalarType dtype) :
        dtype_(dtype) {
        CHECK_FAIL(shape.size() <= MAX_TENSOR_DIM);
        dim_ = shape.size();
        numel_ = 1;
        for (int i = dim_ - 1; i >= 0; i--) {
            stride_[i] = numel_;
            numel_ *= shape[i];
            shape_[i] = shape[i];
        }
    }
    Tensor(const Tensor &other) :
        dim_(other.dim_), shape_(other.shape_), stride_(other.stride_),
        dtype_(other.dtype_), numel_(other.numel_),
        storage_(other.storage_) {
    }
    Tensor() :
        storage_() {
    }
    bool defined() const {
        return storage_.get() != nullptr;
    }
    uint64_t numel() const {
        return numel_;
    }
    int dim() const {
        return dim_;
    }
    int device() const {
        return storage_.get()->device();
    }
    uint64_t shape(int d) const {
        return shape_[d];
    }
    uint64_t stride(int d) const {
        return stride_[d];
    }
    void *data_ptr() const {
        return storage_.ptr()->data_ptr();
    }
    size_t storage_bytes() const {
        return storage_.ptr()->size();
    }
    size_t storage_ref_count() const {
        return storage_.ref_count();
    }
    intrusive_ptr<memory::TensorStorage> storage() const {
        return storage_;
    }
};

Tensor empty(std::vector<uint64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    return output;
}

Tensor zeros(std::vector<uint64_t> shape, ScalarType dtype, int device) {
    Tensor output(shape, dtype);
    output.new_storage_(device);
    GpuLauncher::GetInstance()->memset(output.data_ptr(), 0, output.storage_bytes());
    return output;
}

} // namespace pmkl