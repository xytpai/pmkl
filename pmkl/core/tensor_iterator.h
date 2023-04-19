#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <limits>
#include <memory>
#include <algorithm>

#include "tensor.h"
#include "exception.h"

namespace pmkl {

struct SplitUntil32Bit;

class TensorIterator final {
    enum {
        MAX_TENSORS = 8,
    };

private:
    Tensor *tensors_[MAX_TENSORS];
    size_t ptr_offsets_[MAX_TENSORS];
    int64_t shape_[MAX_TENSOR_DIMS];
    int64_t stride_bytes_[MAX_TENSORS][MAX_TENSOR_DIMS];
    int64_t perm_[MAX_TENSOR_DIMS];
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    int num_tensors_ = 0;
    int ndim_ = 0;
    bool is_reordered_ = false;
    bool accumulate_ = false;
    bool final_output_ = true;
    bool is_reduction_ = false;

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
            int64_t sz;
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
            auto element_size_in_bytes = t->element_size_in_bytes();
            for (int i = ndim_ - 1; i >= 0; --i) {
                if (t->shape(i) == 1 && shape_[i] != 1) {
                    stride_bytes_[id][i] = 0;
                } else {
                    stride_bytes_[id][i] = t->stride(i) * element_size_in_bytes;
                }
            }
        }
    }

    void permute_dimensions() {
        int64_t shape_temp[MAX_TENSOR_DIMS];
        int64_t strides_temp[MAX_TENSORS][MAX_TENSOR_DIMS];
        for (int i = 0; i < ndim_; ++i)
            shape_temp[i] = shape_[i];
        for (int i = 0; i < num_tensors_; ++i)
            for (int j = 0; j < ndim_; ++j)
                strides_temp[i][j] = stride_bytes_[i][j];
        for (int i = 0; i < ndim_; ++i)
            shape_[i] = shape_temp[perm_[i]];
        for (int i = 0; i < num_tensors_; ++i) {
            if (!tensors_[i]->defined()) continue;
            for (int j = 0; j < ndim_; ++j)
                stride_bytes_[i][j] = strides_temp[i][perm_[j]];
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
                int64_t stride0 = tensors_[arg]->stride(dim0);
                int64_t stride1 = tensors_[arg]->stride(dim1);
                if (stride0 == 0 || stride1 == 0) {
                    // move on to the next input if one of the dimensions is broadcasted
                    continue;
                } else if (stride0 < stride1) {
                    return -1;
                } else if (stride0 > stride1) {
                    return 1;
                } else {
                    // for equal strides, the dimension with smaller size goes front
                    auto t_dim0 = shape_[dim0];
                    auto t_dim1 = shape_[dim1];
                    // return only if dimensions should be swapped, otherwise move on to the next tensor
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
                    int64_t shape[MAX_TENSOR_DIMS];
                    for (int k = 0; k < ndim_; ++k)
                        shape[perm_[k]] = shape_[k];
                    *tensors_[i] = std::move(empty(shape, ndim_, dtype, device, false));
                }
                auto &stride = tensors_[i]->stride();
                for (int d = 0; d < ndim_; ++d) {
                    stride_bytes_[i][d] = stride[ndim_ - 1 - d] * element_size(dtype);
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
                auto stride0 = stride_bytes_[i][dim0];
                auto stride1 = stride_bytes_[i][dim1];
                if (shape0 * stride0 != stride1) {
                    return false;
                }
            }
            return true;
        };

        // replace each operands stride at dim0 with its stride at dim1
        auto replace_stride = [&](int dim0, int dim1) {
            for (int i = 0; i < num_tensors_; ++i) {
                stride_bytes_[i][dim0] = stride_bytes_[i][dim1];
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
        ptr_offsets_[num_tensors_] = 0;
        tensors_[num_tensors_++] = &output;
        num_outputs_++;
        return *this;
    }
    TensorIterator &add_input(Tensor &input) {
        ptr_offsets_[num_tensors_] = 0;
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

    int64_t shape(int dim) const {
        return shape_[dim];
    }

    int64_t *shape() {
        return shape_;
    }

    int64_t stride_bytes(int arg, int dim) const {
        return stride_bytes_[arg][dim];
    }

    int64_t *strides(int arg) {
        return stride_bytes_[arg];
    }

    int dim() const {
        return ndim_;
    }

    int ndim() const {
        return ndim_;
    }

    int64_t perm(int dim) const {
        return perm_[dim];
    }

    Tensor &outputs(int arg) {
        return *tensors_[arg];
    }

    int64_t numel() const {
        int64_t numel = 1;
        for (int i = 0; i < ndim_; ++i) {
            numel *= shape_[i];
        }
        return numel;
    }

    bool can_use_32bit_indexing() const {
        int64_t max_value = std::numeric_limits<int32_t>::max();
        if (numel() > max_value) {
            return false;
        }
        for (int i = 0; i < num_tensors_; ++i) {
            int64_t max_offset = 1;
            for (int d = 0; d < ndim(); ++d) {
                max_offset += (shape_[d] - 1) * stride_bytes_[i][d];
            }
            if (max_offset > max_value) {
                return false;
            }
        }
        return true;
    }

    bool has_contiguous_first_dim() const {
        for (int i = 0; i < num_tensors_; ++i) {
            if (stride_bytes_[i][0] != tensors_[i]->element_size_in_bytes()) {
                return false;
            }
        }
        return true;
    }

    bool is_contiguous() const {
        if (numel() == 1) {
            return true;
        }
        if (ndim() != 1) {
            return false;
        }
        return has_contiguous_first_dim();
    }

    /// If the kernel should accumulate into the output. Only relevant for reductions
    bool should_accumulate() const {
        return accumulate_;
    }

    bool is_final_output() const {
        return final_output_;
    }

    int get_dim_to_split() const {
        CHECK_FAIL(ndim() >= 1);
        int64_t max_extent = -1;
        int dim_to_split = -1;
        for (int dim = ndim() - 1; dim >= 0; dim--) {
            const int64_t size = shape_[dim];
            if (size == 0) {
                continue;
            }
            for (int i = 0; i < num_tensors_; ++i) {
                // std::abs is necessary to handle some special cases where we support negative strides
                const int64_t extent = (size - 1) * std::abs(stride_bytes_[i][dim]);
                if (extent > max_extent) {
                    max_extent = extent;
                    dim_to_split = dim;
                }
            }
        }
        CHECK_FAIL(max_extent >= 0);
        return dim_to_split;
    }

    bool is_dim_reduced(int dim) const {
        for (int i = 0; i < num_tensors_; ++i) {
            if ((i < num_outputs_) && stride_bytes_[i][dim] == 0 && shape_[dim] > 1) {
                return true;
            }
        }
        return false;
    }

    void narrow(int dim, int64_t start, int64_t size) {
        CHECK_FAIL(dim < ndim() && size >= 1);
        shape_[dim] = size;
        for (int i = 0; i < num_tensors_; ++i) {
            ptr_offsets_[i] += stride_bytes_[i][dim] * start;
        }
        if (size == 1 && !is_reduction_) {
            coalesce_dimensions();
        }
    }

    std::unique_ptr<TensorIterator> split(int dim) {
        CHECK_FAIL(dim >= 0 && dim < ndim() && shape()[dim] >= 2);
        std::unique_ptr<TensorIterator> copy(new TensorIterator(*this));
        bool overlaps = is_dim_reduced(dim);
        auto copy_size = shape_[dim] / 2;
        auto this_size = shape_[dim] - copy_size;
        copy->narrow(dim, 0, copy_size);
        copy->final_output_ &= !overlaps;
        this->narrow(dim, copy_size, this_size);
        this->accumulate_ |= overlaps;
        return copy;
    }

    ScalarType input_dtype(int arg = 0) const {
        return tensors_[num_outputs_ + arg]->dtype();
    }
    ScalarType dtype(int arg = 0) const {
        return tensors_[arg]->dtype();
    }
    void *data_ptr(int arg) const {
        return (void *)((char *)tensors_[arg]->data_ptr() + ptr_offsets_[arg]);
    }
    int64_t element_size_in_bytes(int arg) const {
        return tensors_[arg]->element_size_in_bytes();
    }

    SplitUntil32Bit with_32bit_indexing() const;
};

struct SplitUntil32Bit {
    struct iterator {
        iterator() {
        }
        iterator(const TensorIterator &iter) {
            vec.emplace_back(new TensorIterator(iter));
            vec.emplace_back(nullptr); // ++ first pops the last element
            ++(*this);
        }
        iterator(iterator &&) = default;
        TensorIterator &operator*() const {
            return *vec.back();
        }
        iterator &operator++() {
            vec.pop_back();
            while (!vec.empty() && !vec.back()->can_use_32bit_indexing()) {
                auto &iter = *vec.back();
                int64_t split_dim = iter.get_dim_to_split();
                vec.emplace_back(iter.split(split_dim));
            }
            return *this;
        }
        bool operator==(const iterator &other) const {
            // two iterators are equal if they are the same object or they're both empty
            return this == &other || (vec.empty() && other.vec.empty());
        }
        // needed for C++11 range-based for loop
        bool operator!=(const iterator &other) const {
            return !(*this == other);
        }
        /// stack of TensorIterators to be split
        std::vector<std::unique_ptr<TensorIterator>> vec;
    };
    SplitUntil32Bit(const TensorIterator &iter) :
        iter(iter) {
    }
    iterator begin() const {
        return iterator(iter);
    }
    iterator end() const {
        return iterator();
    }

private:
    const TensorIterator &iter;
};

SplitUntil32Bit TensorIterator::with_32bit_indexing() const {
    return SplitUntil32Bit(*this);
}

std::ostream &operator<<(std::ostream &os, const TensorIterator &iter) {
    os << "TensorIterator(\n\tshape=[";
    for (int i = 0; i < iter.dim(); ++i)
        os << iter.shape(i) << ",";
    os << "\b],\n\t";
    for (int i = 0; i < iter.ntensors(); ++i) {
        os << "stride_bytes_" << i << "=[";
        for (int j = 0; j < iter.dim(); ++j)
            os << iter.stride_bytes(i, j) << ",";
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
