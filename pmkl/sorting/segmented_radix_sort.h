#pragma once

#include "sorting_common.h"
#include "block_radix_sort.h"
#include "exception.h"

using namespace pmkl::utils;

namespace pmkl {
namespace sorting {

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_segments, int num_elements) {
    using SortMethod = SegmentedBlockRadixSort<key_t, value_t, IS_DESCENDING, BLOCK_THREADS, KEYS_PER_ITEM,
                                               false>;
    using KeyTraitsT = typename SortMethod::KeyTraitsT;
    auto l = GpuLauncher::GetInstance();
    l->submit(
        SortMethod::GetSharedLocalMemorySize(),
        {num_segments}, {BLOCK_THREADS},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int offset = b * num_elements;
            auto method = SortMethod(info);
            KeyTraitsT pkey[SortMethod::REG_LEN];
            value_t pvalue[SortMethod::REG_LEN];
            method.read_key_from_global(pkey, keys_in + offset, num_elements);
            method.read_value_from_global(pvalue, values_in + offset, num_elements);
            method.sort(pkey, pvalue);
            method.write_key_to_global(keys_out + offset, pkey, num_elements);
            method.write_value_to_global(values_out + offset, pvalue, num_elements);
        });
    if (l->is_sync_mode()) l->stream_sync();
}

template <typename key_t, typename value_t, bool IS_DESCENDING>
inline void segmented_sort_pairs_(const key_t *keys_in, key_t *keys_out,
                                  const value_t *values_in, value_t *values_out,
                                  int num_segments, int num_elements) {
    if (num_elements > 4096) {
        CHECK_FAIL(false, "num_elements should shorter than 4096");
    } else if (num_elements > 2048) {
        segmented_sort_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 1024>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 1024) {
        segmented_sort_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 512) {
        segmented_sort_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 256>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else if (num_elements > 256) {
        segmented_sort_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 128>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    } else {
        segmented_sort_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 64>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements);
    }
}

template <typename key_t, typename value_t>
void segmented_sort_pairs(const key_t *keys_in, key_t *keys_out,
                          const value_t *values_in, value_t *values_out,
                          int num_segments, int num_elements, bool descending) {
    if (descending)
        segmented_sort_pairs_<key_t, value_t, true>(
            keys_in, keys_out, values_in, values_out,
            num_segments, num_elements);
    else
        segmented_sort_pairs_<key_t, value_t, false>(
            keys_in, keys_out, values_in, values_out,
            num_segments, num_elements);
}

}
} // namespace pmkl::sorting