#pragma once

#include <limits>

#include "sorting_common.h"
#include "block_radix_processer.h"
#include "core.h"

using namespace pmkl::utils;

namespace pmkl {
namespace sorting {

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_segments, int num_elements) {
    using SortMethod = BlockRadixProcesser<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    auto l = GpuLauncher::GetInstance();
    auto padding_key = IS_DESCENDING ? std::numeric_limits<key_t>::lowest() : std::numeric_limits<key_t>::max();
    l->submit(
        SortMethod::GetSharedLocalMemorySize(),
        {num_segments}, {BLOCK_THREADS},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int b_offset = b * num_elements;
            int lid = info.thread_idx(0);
            auto method = SortMethod(info);

            key_t keys[SortMethod::REG_LEN];
            value_t values[SortMethod::REG_LEN];

#pragma unroll
            for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
                int offset = lid * KEYS_PER_ITEM + ITEM;
                if (offset < num_elements) {
                    keys[ITEM] = keys_in[b_offset + offset];
                    values[ITEM] = values_in[b_offset + offset];
                } else {
                    keys[ITEM] = padding_key;
                }
            }

            method.sort_blocked(keys, values, 0, sizeof(key_t) * 8);

#pragma unroll
            for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
                int offset = lid * KEYS_PER_ITEM + ITEM;
                if (offset < num_elements) {
                    keys_out[b_offset + offset] = keys[ITEM];
                    values_out[b_offset + offset] = values[ITEM];
                }
            }
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

template <typename key_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_sort_impl(
    const key_t *keys_in, key_t *keys_out,
    int num_segments, int num_elements) {
    using SortMethod = BlockRadixProcesser<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING>;
    auto l = GpuLauncher::GetInstance();
    auto padding_key = IS_DESCENDING ? std::numeric_limits<key_t>::lowest() : std::numeric_limits<key_t>::max();
    l->submit(
        SortMethod::GetSharedLocalMemorySize(),
        {num_segments}, {BLOCK_THREADS},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int b_offset = b * num_elements;
            int lid = info.thread_idx(0);
            auto method = SortMethod(info);

            key_t keys[SortMethod::REG_LEN];

#pragma unroll
            for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
                int offset = lid * KEYS_PER_ITEM + ITEM;
                if (offset < num_elements) {
                    keys[ITEM] = keys_in[b_offset + offset];
                } else {
                    keys[ITEM] = padding_key;
                }
            }

            method.sort_blocked(keys, 0, sizeof(key_t) * 8);

#pragma unroll
            for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
                int offset = lid * KEYS_PER_ITEM + ITEM;
                if (offset < num_elements) {
                    keys_out[b_offset + offset] = keys[ITEM];
                }
            }
        });
    if (l->is_sync_mode()) l->stream_sync();
}

template <typename key_t, bool IS_DESCENDING>
inline void segmented_sort_(const key_t *keys_in, key_t *keys_out,
                            int num_segments, int num_elements) {
    if (num_elements > 4096) {
        CHECK_FAIL(false, "num_elements should shorter than 4096");
    } else if (num_elements > 2048) {
        segmented_sort_impl<key_t, IS_DESCENDING, 4, 1024>(
            keys_in, keys_out, num_segments, num_elements);
    } else if (num_elements > 1024) {
        segmented_sort_impl<key_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, num_segments, num_elements);
    } else if (num_elements > 512) {
        segmented_sort_impl<key_t, IS_DESCENDING, 4, 256>(
            keys_in, keys_out, num_segments, num_elements);
    } else if (num_elements > 256) {
        segmented_sort_impl<key_t, IS_DESCENDING, 4, 128>(
            keys_in, keys_out, num_segments, num_elements);
    } else {
        segmented_sort_impl<key_t, IS_DESCENDING, 4, 64>(
            keys_in, keys_out, num_segments, num_elements);
    }
}

template <typename key_t>
void segmented_sort(const key_t *keys_in, key_t *keys_out,
                    int num_segments, int num_elements, bool descending) {
    if (num_segments * num_elements == 0) return;
    if (descending)
        segmented_sort_<key_t, true>(
            keys_in, keys_out,
            num_segments, num_elements);
    else
        segmented_sort_<key_t, false>(
            keys_in, keys_out,
            num_segments, num_elements);
}

}
} // namespace pmkl::sorting