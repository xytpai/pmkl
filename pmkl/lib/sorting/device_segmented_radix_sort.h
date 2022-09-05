#pragma once

#include "common.h"

template <typename key_t, typename value_t, typename OffsetIteratorT>
inline void segmented_sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t num_elements, int64_t num_segments,
    OffsetIteratorT begin_offsets, OffsetIteratorT end_offsets,
    bool descending = false, int64_t begin_bit = 0, int64_t end_bit = sizeof(key_t) * 8) {
    TORCH_CHECK(num_elements <= std::numeric_limits<int>::max(),
                "cub sort does not support sorting more than INT_MAX elements");
    TORCH_CHECK(num_segments <= std::numeric_limits<int>::max(),
                "cub sort does not support sorting more than INT_MAX elements");
    using key_t_ = typename detail::cuda_type<key_t>::type;

    auto allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr keys_out_owner;

    if (keys_out == nullptr) {
        keys_out_owner = allocator->allocate(num_elements * sizeof(key_t));
        keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
    }

    const key_t_ *keys_in_ = reinterpret_cast<const key_t_ *>(keys_in);
    key_t_ *keys_out_ = reinterpret_cast<key_t_ *>(keys_out);

    if (descending) {
        CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairsDescending,
                    keys_in_, keys_out_, values_in, values_out,
                    num_elements, num_segments, begin_offsets, end_offsets,
                    begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
    } else {
        CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairs,
                    keys_in_, keys_out_, values_in, values_out,
                    num_elements, num_segments, begin_offsets, end_offsets,
                    begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
    }
}
