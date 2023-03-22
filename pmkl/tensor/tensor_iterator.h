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
    Tensor *tensors_[MAX_TENSORS];
    uint64_t shape_[MAX_TENSOR_DIM];
    uint64_t strides_[MAX_TENSORS][MAX_TENSOR_DIM];
    uint64_t perm_[MAX_TENSOR_DIM];
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    int num_tensors_ = 0;
    int ndim_ = 0;
    bool is_reordered_ = false;

    bool check_and_compute_dim() {
        bool is_first = true;
        int first_dim;
        for (int i = 0; i < num_tensors_; ++i) {
            if (!tensors_[i]->defined()) continue;
            if (is_first) {
                first_dim = tensors_[i]->dim();
                is_first = false;
            } else {
                if (first_dim != tensors_[i]->dim())
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
                if (!tensors_[j]->defined()) continue;
                if (is_first) {
                    sz = tensors_[j]->shape(i);
                    is_first = false;
                } else {
                    auto sz_ = tensors_[j]->shape(i);
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
            if (!t->defined()) continue;
            for (int i = ndim_ - 1; i >= 0; --i) {
                if (t->shape(i) == 1 && shape_[i] != 1) {
                    strides_[id][i] = 0;
                } else {
                    strides_[id][i] = t->stride(i);
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
        for (int i = 0; i < num_tensors_; ++i) {
            if (!tensors_[i]->defined()) continue;
            for (int j = 0; j < ndim_; ++j)
                strides_[i][j] = strides_temp[i][perm_[j]];
        }
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
                if (!tensors_[arg]->defined()) continue;
                uint64_t stride0 = tensors_[arg]->stride(dim0);
                uint64_t stride1 = tensors_[arg]->stride(dim1);
                if (stride0 == 0 || stride1 == 0) {
                    //move on to the next input if one of the dimensions is broadcasted
                    continue;
                } else if (stride0 < stride1) {
                    return -1;
                } else if (stride0 > stride1) {
                    return 1;
                } else {
                    // for equal strides, the dimension with smaller size goes front
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
        is_reordered_ = true;
    }

    void allocate_outputs() {
        auto device = tensors_[num_outputs_]->device();
        auto dtype = tensors_[num_outputs_]->dtype();
        for (int i = 0; i < num_outputs_; ++i) {
            if (!tensors_[i]->defined()) {
                if (!is_reordered_) {
                    *tensors_[i] = std::move(empty(shape_, ndim_, dtype, device, false));
                } else {
                    uint64_t shape[MAX_TENSOR_DIM];
                    for (int k = 0; k < ndim_; ++k)
                        shape[perm_[k]] = shape_[k];
                    *tensors_[i] = std::move(empty(shape, ndim_, dtype, device, false));
                }
                auto &stride = tensors_[i]->stride();
                for (int d = 0; d < ndim_; ++d) {
                    strides_[i][d] = stride[ndim_ - 1 - d];
                }
            }
        }
    }

    void coalesce_dimensions() {
        if (ndim_ <= 1) return;
        // We can coalesce two adjacent dimensions if either dim has size 1 or if:
        // shape[n] * stride[n] == stride[n + 1].
        auto can_coalesce = [&](int dim0, int dim1) {
            auto shape0 = shape_[dim0];
            auto shape1 = shape_[dim1];
            if (shape0 == 1 || shape1 == 1) {
                return true;
            }
            for (int i = 0; i < num_tensors_; ++i) {
                auto stride0 = strides_[i][dim0];
                auto stride1 = strides_[i][dim1];
                if (shape0 * stride0 != stride1) {
                    return false;
                }
            }
            return true;
        };

        // replace each operands stride at dim0 with its stride at dim1
        auto replace_stride = [&](int dim0, int dim1) {
            for (int i = 0; i < num_tensors_; ++i) {
                strides_[i][dim0] = strides_[i][dim1];
            }
        };

        int prev_dim = 0;
        for (int dim = 1; dim < ndim_; ++dim) {
            if (can_coalesce(prev_dim, dim)) {
                if (shape_[prev_dim] == 1) {
                    replace_stride(prev_dim, dim);
                }
                shape_[prev_dim] *= shape_[dim];
            } else {
                prev_dim++;
                if (prev_dim != dim) {
                    replace_stride(prev_dim, dim);
                    shape_[prev_dim] = shape_[dim];
                }
            }
        }

        ndim_ = prev_dim + 1;
    }

public:
    friend std::ostream &operator<<(std::ostream &os, const TensorIterator &iter);
    TensorIterator() {
    }
    TensorIterator &add_output(Tensor &output) {
        tensors_[num_tensors_++] = &output;
        num_outputs_++;
        return *this;
    }
    TensorIterator &add_input(Tensor &input) {
        tensors_[num_tensors_++] = &input;
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
        return *tensors_[arg];
    }

    TensorIterator &build_for_loops() {
        CHECK_FAIL(check_and_compute_dim());
        compute_shape();
        compute_strides();
        reorder_dimensions();
        allocate_outputs();
        coalesce_dimensions();
        return *this;
    }

    int shape(int dim) const {
        return shape_[dim];
    }

    int stride(int arg, int dim) const {
        return strides_[arg][dim];
    }

    int dim() const {
        return ndim_;
    }

    uint64_t perm(int dim) const {
        return perm_[dim];
    }

    Tensor &outputs(int arg) {
        return *tensors_[arg];
    }
};

std::ostream &operator<<(std::ostream &os, const TensorIterator &iter) {
    os << "TensorIterator(\n\tshape=[";
    for (int i = 0; i < iter.dim(); ++i)
        os << iter.shape(i) << ",";
    os << "\b],\n\t";
    for (int i = 0; i < iter.ntensors(); ++i) {
        os << "strides_" << i << "=[";
        for (int j = 0; j < iter.dim(); ++j)
            os << iter.stride(i, j) << ",";
        os << "\b],\n\t";
    }
    os << "perm=[";
    for (int i = 0; i < iter.dim(); ++i)
        os << iter.perm(i) << ",";
    os << "\b],\n\tdim=" << iter.dim() << ",\n\tninputs=" << iter.ninputs();
    os << ",\n\tnoutputs=" << iter.noutputs() << ")";
    return os;
}

} // namespace pmkl
