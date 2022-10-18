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
        PACKED_SCAN_SIZE = COUNTER_LANES * BLOCK_THREADS * sizeof(CounterT),
        DIGIT_BITS = sizeof(DigitT) << 3,
        DIGIT_MASK = (1 << DIGIT_BITS) - 1,
    };

private:
    KernelInfo &info_;
    int lid_;
    char *local_storage_;

public:
    static int GetSharedLocalMemorySize() {
        constexpr int KV_TYPE_SIZE = KEYS_ONLY ? sizeof(KeyT) : std::max(sizeof(KeyT), sizeof(ValueT));
        constexpr int EXCHANGE_SIZE =
            BLOCK_THREADS * KEYS_PER_THREAD * KV_TYPE_SIZE;
        return std::max(EXCHANGE_SIZE, (int)PACKED_SCAN_SIZE);
    }

    DEVICE inline BlockRadixProcesser(KernelInfo &info) :
        info_(info) {
        lid_ = info_.thread_idx(0);
        local_storage_ = reinterpret_cast<char *>(info_.shared_ptr());
    }

    DEVICE inline void convert_keys(KeyTraitsT (&pkeys)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            pkeys[ITEM] = KeyTraits<KeyT>::convert(
                *reinterpret_cast<KeyT *>(&pkeys[ITEM]));
        }
    }

    DEVICE inline void deconvert_keys(KeyTraitsT (&pkeys)[KEYS_PER_THREAD]) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto key = KeyTraits<KeyT>::deconvert(pkeys[ITEM]);
            pkeys[ITEM] = *reinterpret_cast<KeyTraitsT *>(&key);
        }
    }

    template <typename DataT>
    DEVICE inline void exchange(
        DataT (&data)[KEYS_PER_THREAD],
        int (&rank)[KEYS_PER_THREAD]) {
        auto local_storage = reinterpret_cast<DataT *>(local_storage_);
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage[rank[ITEM]] = data[ITEM];
        }
        info_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            data[ITEM] = local_storage[offset];
        }
        info_.barrier();
    }

    template <typename DataT>
    DEVICE inline void exchange(
        DataT (&data)[KEYS_PER_THREAD],
        int (&rank)[KEYS_PER_THREAD],
        int lower_offset,
        int upper_offset) {
        auto local_storage = reinterpret_cast<DataT *>(local_storage_);
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (rank[ITEM] >= lower_offset && rank[ITEM] < upper_offset) {
                local_storage[rank[ITEM] - lower_offset] = data[ITEM];
            }
        }
        info_.barrier();
        int new_length = upper_offset - lower_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < new_length) {
                data[ITEM] = local_storage[offset];
            }
        }
        info_.barrier();
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key, int begin, int pass) {
        return ((key >> begin) & ((1 << pass) - 1));
    }

    DEVICE inline void rank_keys(
        KeyTraitsT (&pkeys)[KEYS_PER_THREAD],
        int (&rank)[KEYS_PER_THREAD],
        int begin_bit,
        int pass_bits) {
        DigitT *digit_counters[KEYS_PER_THREAD];
        DigitT sub_counters[KEYS_PER_THREAD];
        auto scan_storage = reinterpret_cast<CounterT *>(local_storage_);
        auto buckets = reinterpret_cast<DigitT(*)[BLOCK_THREADS][PACKING_RATIO]>(local_storage_);

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            scan_storage[lid_ * COUNTER_LANES + ITEM] = 0;
        }
        info_.barrier();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(pkeys[ITEM], begin_bit, pass_bits);
            auto sub_counter = digit >> LOG_COUNTER_LANES;
            auto counter_lane = digit & (COUNTER_LANES - 1);
            if (IS_DESCENDING) {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }
            sub_counters[ITEM] = sub_counter;
            digit_counters[ITEM] = &buckets[counter_lane][lid_][sub_counter];
            rank[ITEM] = *digit_counters[ITEM];
            *digit_counters[ITEM] = rank[ITEM] + 1;
        }
        info_.barrier();

        CounterT temp = block_aligned_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS>(info_, scan_storage, lid_);

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            temp = temp << DIGIT_BITS;
            c += temp;
        }

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            DigitT cc = (c >> (sub_counters[ITEM] * DIGIT_BITS)) & DIGIT_MASK;
            rank[ITEM] += *digit_counters[ITEM] + cc;
        }
        info_.barrier();
    }

    DEVICE inline void sort_blocked(
        KeyT (&pkeys)[KEYS_PER_THREAD],
        ValueT (&pvalues)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit) {
        KeyTraitsT(&unsigned_keys)[KEYS_PER_THREAD] =
            reinterpret_cast<KeyTraitsT(&)[KEYS_PER_THREAD]>(pkeys);
        convert_keys(unsigned_keys);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int rank[KEYS_PER_THREAD];
            rank_keys(unsigned_keys, rank, begin_bit, pass_bits);
            begin_bit += RADIX_BITS;
            exchange<KeyTraitsT>(unsigned_keys, rank);
            if (!KEYS_ONLY) exchange<ValueT>(pvalues, rank);
            if (begin_bit >= end_bit) break;
        }
        deconvert_keys(unsigned_keys);
    }

    DEVICE inline void sort_blocked(
        KeyT (&pkeys)[KEYS_PER_THREAD],
        int begin_bit,
        int end_bit) {
        KeyTraitsT(&unsigned_keys)[KEYS_PER_THREAD] =
            reinterpret_cast<KeyTraitsT(&)[KEYS_PER_THREAD]>(pkeys);
        convert_keys(unsigned_keys);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int rank[KEYS_PER_THREAD];
            rank_keys(unsigned_keys, rank, begin_bit, pass_bits);
            begin_bit += RADIX_BITS;
            exchange<KeyTraitsT>(unsigned_keys, rank);
            if (begin_bit >= end_bit) break;
        }
        deconvert_keys(unsigned_keys);
    }
};

}
} // namespace pmkl::sorting