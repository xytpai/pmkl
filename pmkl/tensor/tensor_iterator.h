#pragma once

#include <iostream>
#include <ostream>
#include <vector>

#include "tensor.h"
#include "exception.h"

namespace pmkl {

class TensorIterator final {
    enum {
        MAX_TENSORS = 8,
    };

private:
    Tensor tensors_[MAX_TENSORS];
    uint64_t shape_[MAX_TENSOR_DIM];
    uint64_t strides_[MAX_TENSORS][MAX_TENSOR_DIM];
    uint64_t perm_[MAX_TENSOR_DIM];
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    int num_tensors_ = 0;
    int ndim_ = 0;

    bool check_and_compute_dim() {
        bool is_first = true;
        int first_dim;
        for (int i = 0; i < num_tensors_; ++i) {
            if (!tensors_[i].defined()) continue;
            if (is_first) {
                first_dim = tensors_[i].dim();
                is_first = false;
            } else {
                if (first_dim != tensors_[i].dim())
                    return false;
            }
        }
        ndim_ = first_dim;
        return true;
    }

    void compute_shape() {
        for (int i = ndim_ - 1; i >= 0; --i) {
            bool is_first = true;
            uint64_t sz;
            for (int j = 0; j < num_tensors_; ++j) {
                if (!tensors_[j].defined()) continue;
                if (is_first) {
                    sz = tensors_[j].shape(i);
                    is_first = false;
                } else {
                    auto sz_ = tensors_[j].shape(i);
                    CHECK_FAIL(sz == sz_ || sz == 1 || sz_ == 1);
                    sz = sz == 1 ? sz_ : sz;
                }
            }
            shape_[i] = sz;
        }
    }

    void compute_strides() {
        for (int id = 0; id < num_tensors_; ++id) {
            auto t = tensors_[id];
            if (!t.defined()) continue;
            for (int i = ndim_ - 1; i >= 0; --i) {
                if (t.shape(i) == 1 && shape_[i] != 1) {
                    strides_[id][i] = 0;
                } else {
                    strides_[id][i] = t.stride(i);
                }
            }
        }
    }

    void permute_dimensions() {
        uint64_t shape_temp[MAX_TENSOR_DIM];
        uint64_t strides_temp[MAX_TENSORS][MAX_TENSOR_DIM];
        for (int i = 0; i < ndim_; ++i)
            shape_temp[i] = shape_[i];
        for (int i = 0; i < num_tensors_; ++i)
            for (int j = 0; j < ndim_; ++j)
                strides_temp[i][j] = strides_[i][j];
        for (int i = 0; i < ndim_; ++i)
            shape_[i] = shape_temp[perm_[i]];
        for (int i = 0; i < num_tensors_; ++i)
            for (int j = 0; j < ndim_; ++j)
                strides_[i][j] = strides_temp[i][perm_[j]];
    }

    void reorder_dimensions() {
        if (ndim_ == 1) {
            perm_[0] = 0;
            return;
        }
        int ct = 0;
        for (int i = ndim_ - 1; i >= 0; --i) {
            perm_[ct++] = i;
        }

        auto should_swap = [&](size_t dim0, size_t dim1) {
            for (int arg = 0; arg < num_tensors_; ++arg) {
                if (!tensors_[arg].defined()) continue;
                uint64_t stride0 = tensors_[arg].stride(dim0);
                uint64_t stride1 = tensors_[arg].stride(dim1);
                if (stride0 == 0 || stride1 == 0) {
                    //move on to the next input if one of the dimensions is broadcasted
                    continue;
                } else if (stride0 < stride1) {
                    return -1;
                } else if (stride0 > stride1) {
                    return 1;
                } else {
                    auto t_dim0 = shape_[dim0];
                    auto t_dim1 = shape_[dim1];
                    //return only if dimensions should be swapped, otherwise move on to the next tensor
                    if (t_dim0 > t_dim1) {
                        return 1;
                    }
                }
            }
            return 0;
        };

        // insertion sort with support for ambiguous comparisons
        for (int i = 1; i < ndim_; ++i) {
            int dim1 = i;
            for (int dim0 = i - 1; dim0 >= 0; dim0--) {
                int comparison = should_swap(perm_[dim0], perm_[dim1]);
                if (comparison > 0) {
                    std::swap(perm_[dim0], perm_[dim1]);
                    dim1 = dim0;
                } else if (comparison < 0) {
                    break;
                }
            }
        }

        permute_dimensions();
    }

    void allocate_outputs() {
        auto device = tensors_[num_outputs_].device();
        auto dtype = tensors_[num_outputs_].dtype();
        for (int i = 0; i < num_outputs_; ++i) {
            if (!tensors_[i].defined()) {
                tensors_[i] = std::move(empty(shape_, ndim_, dtype, device));
            }
        }
    }

public:
    friend std::ostream &operator<<(std::ostream &os, const TensorIterator &iter);
    TensorIterator() {
    }
    TensorIterator &add_output(const Tensor &output) {
        tensors_[num_tensors_++] = output;
        num_outputs_++;
        return *this;
    }
    TensorIterator &add_input(const Tensor &input) {
        tensors_[num_tensors_++] = input;
        num_inputs_++;
        return *this;
    }
    TensorIterator &add_output(Tensor &&output) = delete;
    TensorIterator &add_input(Tensor &&input) = delete;

    int ntensors() const {
        return num_tensors_;
    }
    int noutputs() const {
        return num_outputs_;
    }
    int ninputs() const {
        return num_inputs_;
    }
    const Tensor &tensor(int arg) const {
        return tensors_[arg];
    }

    TensorIterator &build_for_loops() {
        CHECK_FAIL(check_and_compute_dim());
        compute_shape();
        compute_strides();
        reorder_dimensions();
        allocate_outputs();
        return *this;
    }

    int shape(int dim) const {
        return shape_[dim];
    }

    int stride(int arg, int dim) const {
        return strides_[arg][dim];
    }

    Tensor &outputs(int arg) {
        return tensors_[arg];
    }
};

std::ostream &operator<<(std::ostream &os, const TensorIterator &iter) {
    os << "TensorIterator\nshape:";
    for (int i = 0; i < iter.ndim_; ++i)
        os << iter.shape_[i] << ",";
    for (int i = 0; i < iter.num_tensors_; ++i) {
        os << "\nstrides_" << i << ":";
        for (int j = 0; j < iter.ndim_; ++j)
            os << iter.strides_[i][j] << ",";
    }
    os << "\nperm:";
    for (int i = 0; i < iter.ndim_; ++i)
        os << iter.perm_[i] << ",";
    os << "\n";
    return os;
}

} // namespace pmkl
