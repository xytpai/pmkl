#pragma once

#include <limits>

#include "sorting_common.h"
#include "block_radix_processer.h"
#include "exception.h"
#include "segmented_radix_sort.h"

using namespace pmkl::utils;

namespace pmkl {
namespace sorting {

template <typename key_t, typename value_t, bool IS_DESCENDING, int KEYS_PER_ITEM, int BLOCK_THREADS>
void segmented_select_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int num_segments, int num_elements, int num_topk) {
    using SelectMethod = BlockRadixProcesser<key_t, BLOCK_THREADS, KEYS_PER_ITEM, IS_DESCENDING, value_t>;
    CHECK_FAIL(num_topk <= SelectMethod::PROCESSING_LENGTH, "num_topk should smaller than PROCESSING_LENGTH");
    auto l = GpuLauncher::GetInstance();
    auto padding_key = IS_DESCENDING ? std::numeric_limits<key_t>::lowest() : std::numeric_limits<key_t>::max();
    l->submit(
        SelectMethod::GetSharedLocalMemorySize() + num_topk * (sizeof(key_t) + sizeof(value_t)),
        {num_segments}, {BLOCK_THREADS},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int b_offset = b * num_elements;
            int lid = info.thread_idx(0);
            auto method = SelectMethod(info);

            key_t keys[SelectMethod::REG_LEN];
            value_t values[SelectMethod::REG_LEN];

            key_t *keys_temp =
                reinterpret_cast<key_t *>(info.shared_ptr() + SelectMethod::GetSharedLocalMemorySize());
            value_t *values_temp =
                reinterpret_cast<value_t *>(
                    info.shared_ptr() + SelectMethod::GetSharedLocalMemorySize() + num_topk * sizeof(key_t));

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

            int num_start = SelectMethod::PROCESSING_LENGTH;
            while (num_start < num_elements) {
                method.select_blocked(keys, values, sizeof(key_t) * 8, 0,
                                      num_topk, keys_temp, values_temp);
                info.barrier();
#pragma unroll
                for (int ITEM = 0; ITEM < KEYS_PER_ITEM; ++ITEM) {
                    int offset = lid * KEYS_PER_ITEM + ITEM;
                    if (offset < num_topk) {
                        keys[ITEM] = keys_temp[offset];
                        values[ITEM] = values_temp[offset];
                    } else {
                        offset += num_start - num_topk;
                        if (offset < num_elements) {
                            keys[ITEM] = keys_in[b_offset + offset];
                            values[ITEM] = values_in[b_offset + offset];
                        } else {
                            keys[ITEM] = padding_key;
                        }
                    }
                }
                num_start += SelectMethod::PROCESSING_LENGTH - num_topk;
                info.barrier();
            }

            method.select_blocked(keys, values, sizeof(key_t) * 8, 0,
                                  num_topk, keys_out + b_offset, values_out + b_offset);
        });
    if (l->is_sync_mode()) l->stream_sync();
}

template <typename key_t, typename value_t, bool IS_DESCENDING>
inline void segmented_select_pairs_(const key_t *keys_in, key_t *keys_out,
                                    const value_t *values_in, value_t *values_out,
                                    int num_segments, int num_elements, int num_topk) {
    if (num_elements > 2048) {
        segmented_select_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 1024>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, num_topk);
    } else if (num_elements > 1024) {
        segmented_select_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 512>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, num_topk);
    } else if (num_elements > 512) {
        segmented_select_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 256>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, num_topk);
    } else if (num_elements > 256) {
        segmented_select_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 128>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, num_topk);
    } else {
        segmented_select_pairs_impl<key_t, value_t, IS_DESCENDING, 4, 64>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, num_topk);
    }
}

template <typename key_t, typename value_t>
void segmented_select_pairs(const key_t *keys_in, key_t *keys_out,
                            const value_t *values_in, value_t *values_out,
                            int num_segments, int num_elements, int num_topk, bool is_descending) {
    if (num_segments * num_elements == 0) return;
    if (num_topk >= num_elements || num_topk > 256 || num_topk > (int)(0.5 * num_elements)) {
        segmented_sort_pairs<key_t, value_t>(
            keys_in, keys_out, values_in, values_out, num_segments, num_elements, is_descending);
    } else {
        if (is_descending)
            segmented_select_pairs_<key_t, value_t, true>(
                keys_in, keys_out, values_in, values_out,
                num_segments, num_elements, num_topk);
        else
            segmented_select_pairs_<key_t, value_t, false>(
                keys_in, keys_out, values_in, values_out,
                num_segments, num_elements, num_topk);
    }
}

}
} // namespace pmkl::sorting
