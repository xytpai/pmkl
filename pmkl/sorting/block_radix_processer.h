#pragma once

#include "sorting_common.h"

namespace pmkl {
namespace sorting {

template <
    typename KeyT,
    int BLOCK_THREADS = 128,
    int KEYS_PER_THREAD = 4,
    bool IS_DESCENDING = false,
    typename ValueT = NullType,
    typename DigitT = uint16_t,   // Covering BLOCK_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform packed prefix sum.
    int RADIX_BITS = 4>
class BlockRadixProcesser {
public:
    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
    static_assert(
        ((1l << (sizeof(DigitT) << 3)) - 1)
            >= (BLOCK_THREADS * KEYS_PER_THREAD),
        " ");
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;

    enum {
        PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_THREAD,
        REG_LEN = KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
        LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
        DIGIT_BITS = sizeof(DigitT) << 3,
    };

private:
    union RankT {
        CounterT counters[COUNTER_LANES][BLOCK_THREADS];
        CounterT counters_flat[COUNTER_LANES * BLOCK_THREADS];
        DigitT buckets[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
    };

    union LocalStorage {
        RankT rank_storage;
        KeyTraitsT exchange_ukeys[PROCESSING_LENGTH];
        ValueT exchange_values[PROCESSING_LENGTH];
        int valid_items[BLOCK_THREADS];
    };

    KernelInfo &info_;
    LocalStorage &local_storage_;
    int lid_;

public:
    static HOST_DEVICE int GetSharedLocalMemorySize() {
        return sizeof(LocalStorage);
    }

    DEVICE inline BlockRadixProcesser(KernelInfo &info) :
        info_(info), lid_(info.thread_idx(0)),
        local_storage_(reinterpret_cast<LocalStorage &>(*info.shared_ptr())) {
    }

    DEVICE inline void convert_keys(KeyTraitsT (&ukeys)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ukeys[ITEM] = KeyTraits<KeyT>::convert(
                *reinterpret_cast<KeyT *>(&ukeys[ITEM]));
        }
    }

    DEVICE inline void deconvert_keys(KeyTraitsT (&ukeys)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto ukey = KeyTraits<KeyT>::deconvert(ukeys[ITEM]);
            ukeys[ITEM] = *reinterpret_cast<KeyTraitsT *>(&ukey);
        }
    }

    DEVICE inline void exchange_keys(
        KeyTraitsT (&ukeys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_ukeys[ranks[ITEM]] = ukeys[ITEM];
        }
        info_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            ukeys[ITEM] = local_storage_.exchange_ukeys[offset];
        }
        info_.barrier();
    }

    DEVICE inline void exchange_values(
        ValueT (&values)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_.exchange_values[ranks[ITEM]] = values[ITEM];
        }
        info_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            values[ITEM] = local_storage_.exchange_values[offset];
        }
        info_.barrier();
    }

    DEVICE inline void exchange_keys(
        KeyTraitsT (&ukeys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int lower_offset,
        int upper_offset,
        uint32_t *mask) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks[ITEM] >= lower_offset && ranks[ITEM] < upper_offset) {
                local_storage_.exchange_ukeys[ranks[ITEM] - lower_offset] = ukeys[ITEM];
            }
        }
        info_.barrier();
        *mask = 0u;
        int new_length = upper_offset - lower_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < new_length) {
                *mask |= (1u << ITEM);
                ukeys[ITEM] = local_storage_.exchange_ukeys[offset];
            }
        }
        info_.barrier();
    }

    DEVICE inline void exchange_values(
        ValueT (&values)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int lower_offset,
        int upper_offset) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks[ITEM] >= lower_offset && ranks[ITEM] < upper_offset) {
                local_storage_.exchange_values[ranks[ITEM] - lower_offset] = values[ITEM];
            }
        }
        info_.barrier();
        int new_length = upper_offset - lower_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < new_length) {
                values[ITEM] = local_storage_.exchange_values[offset];
            }
        }
        info_.barrier();
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key, int begin, int pass) {
        return ((key >> begin) & ((1 << pass) - 1));
    }

    DEVICE inline void rank_keys(
        KeyTraitsT (&ukeys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int begin_bit,
        int pass_bits) {
        DigitT *digit_counters[KEYS_PER_THREAD];

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            local_storage_.rank_storage.counters[ITEM][lid_] = 0;
        }
        info_.barrier();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(ukeys[ITEM], begin_bit, pass_bits);
            auto sub_counter = digit >> LOG_COUNTER_LANES;
            auto counter_lane = digit & (COUNTER_LANES - 1);
            if (IS_DESCENDING) {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }
            digit_counters[ITEM] =
                &local_storage_.rank_storage.buckets[counter_lane][lid_][sub_counter];
            ranks[ITEM] = *digit_counters[ITEM];
            *digit_counters[ITEM] = ranks[ITEM] + 1;
        }
        info_.barrier();

        CounterT exclusive = block_aligned_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS>(
            info_,
            local_storage_.rank_storage.counters_flat,
            lid_);

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            exclusive = exclusive << DIGIT_BITS;
            c += exclusive;
        }

#pragma unroll
        for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
            local_storage_.rank_storage.counters[INDEX][lid_] += c;
        }
        info_.barrier();

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks[ITEM] += *digit_counters[ITEM];
        }
        info_.barrier();
    }

    DEVICE inline void find_select_offset(
        int carry,
        int num_to_select,
        int *out_offset_select,
        int *out_offset_active) {
        *out_offset_select = 0;
        *out_offset_active = 0;
#pragma unroll
        for (int DIGIT = 1; DIGIT < RADIX_BUCKETS; ++DIGIT) {
            auto sub_counter = DIGIT >> LOG_COUNTER_LANES;
            auto counter_lane = DIGIT & (COUNTER_LANES - 1);
            auto count = (int)(local_storage_.rank_storage.buckets[counter_lane][0][sub_counter]);
            if (count > num_to_select) {
                *out_offset_active = count;
                break;
            }
            *out_offset_select = count;
        }
        if (*out_offset_active == 0) *out_offset_active = carry;
    }

    DEVICE inline void rank_keys(
        KeyTraitsT (&ukeys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int begin_bit,
        int pass_bits,
        uint32_t active_mask,
        int num_to_select,
        int *out_offset_select,
        int *out_offset_active) {
        DigitT *digit_counters[KEYS_PER_THREAD];

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            local_storage_.rank_storage.counters[ITEM][lid_] = 0;
        }
        info_.barrier();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks[ITEM] = PROCESSING_LENGTH;
            if (active_mask >> ITEM & 1) {
                auto digit = extract_digit(ukeys[ITEM], begin_bit, pass_bits);
                auto sub_counter = digit >> LOG_COUNTER_LANES;
                auto counter_lane = digit & (COUNTER_LANES - 1);
                if (IS_DESCENDING) {
                    sub_counter = PACKING_RATIO - 1 - sub_counter;
                    counter_lane = COUNTER_LANES - 1 - counter_lane;
                }
                digit_counters[ITEM] =
                    &local_storage_.rank_storage.buckets[counter_lane][lid_][sub_counter];
                ranks[ITEM] = *digit_counters[ITEM];
                *digit_counters[ITEM] = ranks[ITEM] + 1;
            }
        }
        info_.barrier();

        CounterT exclusive = block_aligned_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS>(
            info_,
            local_storage_.rank_storage.counters_flat,
            lid_);

        DigitT *exclusive_ = reinterpret_cast<DigitT *>(&exclusive);
        int carry = 0;
#pragma unroll
        for (int STEP = 0; STEP < PACKING_RATIO; ++STEP) {
            carry += exclusive_[STEP];
        }

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            exclusive = exclusive << DIGIT_BITS;
            c += exclusive;
        }

#pragma unroll
        for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
            local_storage_.rank_storage.counters[INDEX][lid_] += c;
        }
        info_.barrier();

        find_select_offset(carry, num_to_select, out_offset_select, out_offset_active);

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (active_mask >> ITEM & 1) {
                ranks[ITEM] += *digit_counters[ITEM];
            }
        }
        info_.barrier();
    }

    DEVICE inline void sort_blocked(
        KeyT (&keys)[KEYS_PER_THREAD],
        ValueT (&values)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit) {
        KeyTraitsT(&ukeys)[KEYS_PER_THREAD] =
            reinterpret_cast<KeyTraitsT(&)[KEYS_PER_THREAD]>(keys);
        convert_keys(ukeys);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int ranks[KEYS_PER_THREAD];
            rank_keys(ukeys, ranks, begin_bit, pass_bits);
            begin_bit += RADIX_BITS;
            exchange_keys(ukeys, ranks);
            if (!KEYS_ONLY) exchange_values(values, ranks);
            if (begin_bit >= end_bit) break;
        }
        deconvert_keys(ukeys);
    }

    DEVICE inline void sort_blocked(
        KeyT (&keys)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit) {
        KeyTraitsT(&ukeys)[KEYS_PER_THREAD] =
            reinterpret_cast<KeyTraitsT(&)[KEYS_PER_THREAD]>(keys);
        convert_keys(ukeys);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int ranks[KEYS_PER_THREAD];
            rank_keys(ukeys, ranks, begin_bit, pass_bits);
            begin_bit += RADIX_BITS;
            exchange_keys(ukeys, ranks);
            if (begin_bit >= end_bit) break;
        }
        deconvert_keys(ukeys);
    }

    DEVICE inline void store_keys(
        KeyTraitsT *out,
        KeyTraitsT (&ukeys)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int offset_select,
        int num_selected) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks[ITEM] < offset_select) {
                auto ukey = KeyTraits<KeyT>::deconvert(ukeys[ITEM]);
                out[num_selected + ranks[ITEM]] = *reinterpret_cast<KeyTraitsT *>(&ukey);
            }
        }
    }

    DEVICE inline void store_values(
        ValueT *out,
        ValueT (&values)[KEYS_PER_THREAD],
        int (&ranks)[KEYS_PER_THREAD],
        int offset_select,
        int num_selected) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks[ITEM] < offset_select) {
                out[num_selected + ranks[ITEM]] = values[ITEM];
            }
        }
    }

    DEVICE inline void select_blocked(
        KeyT (&keys)[KEYS_PER_THREAD],
        ValueT (&values)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit,
        int num_topk,
        KeyT *out_keys,
        ValueT *out_values) {
        KeyTraitsT(&ukeys)[KEYS_PER_THREAD] =
            reinterpret_cast<KeyTraitsT(&)[KEYS_PER_THREAD]>(keys);
        KeyTraitsT *out_ukeys = reinterpret_cast<KeyTraitsT *>(out_keys);
        convert_keys(ukeys);
        uint32_t active_mask = 0xffffffff;
        int num_selected = 0;
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, begin_bit - end_bit);
            begin_bit -= pass_bits;
            int ranks[KEYS_PER_THREAD];
            int offset_select, offset_active;
            rank_keys(ukeys, ranks, begin_bit, pass_bits,
                      active_mask, num_topk - num_selected,
                      &offset_select, &offset_active);
            if (begin_bit == end_bit) offset_select = num_topk - num_selected;
            if (offset_select > 0) {
                store_keys(out_ukeys, ukeys, ranks, offset_select, num_selected);
                if (!KEYS_ONLY)
                    store_values(out_values, values, ranks, offset_select, num_selected);
            }
            num_selected += offset_select;
            if (num_selected == num_topk) break;
            exchange_keys(ukeys, ranks, offset_select, offset_active, &active_mask);
            if (!KEYS_ONLY) exchange_values(values, ranks, offset_select, offset_active);
        }
    }

    DEVICE inline void select_blocked(
        KeyT (&keys)[KEYS_PER_THREAD],
        ValueT (&values)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit,
        int num_topk,
        KeyT threshold,
        KeyT *out_keys,
        ValueT *out_values,
        int *out_num_valids) {
        int num_local_valids = 0;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (keys[ITEM] >= threshold) num_local_valids++;
        }

        local_storage_.valid_items[lid_] = num_local_valids;
        info_.barrier();

        int num_block_valids = block_aligned_exclusive_cumsum<
            int,
            1,
            BLOCK_THREADS>(
            info_,
            local_storage_.valid_items,
            lid_);

        int offset = local_storage_.valid_items[lid_];
        info_.barrier();

        if (num_block_valids == 0) {
            *out_num_valids = 0;
        } else if (num_block_valids <= num_topk) {
#pragma unroll
            for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
                if (keys[ITEM] >= threshold) {
                    out_keys[offset] = keys[ITEM];
                    out_values[offset] = values[ITEM];
                    offset++;
                }
            }
            *out_num_valids = num_block_valids;
        } else {
            *out_num_valids = num_topk;
            select_blocked(keys, values, begin_bit,
                           end_bit, num_topk, out_keys, out_values);
        }
    }
};

}
} // namespace pmkl::sorting
