#pragma once

#include "common.h"
#include "key_traits.h"
#include "block_aligned_cumsum.h"

namespace pmkl {

/*
SegmentedBlockRadixSort aim to sort segmented paris efficiently.
sort length is limited to BLOCK_THREADS * KEYS_PER_THREAD
*/
template <
    typename KeyT,
    typename ValueT = NullType,
    bool IS_DESCENDING = false,
    int BLOCK_THREADS = 128,
    int KEYS_PER_THREAD = 4,
    bool USE_INDICES_AS_VALUE = false,
    typename DigitT = uint16_t,   // Covering BLOCK_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform packed prefix sum.
    int RADIX_BITS = 4>
class SegmentedBlockRadixSort {
public:
    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;

    enum {
        SORT_LEN = BLOCK_THREADS * KEYS_PER_THREAD,
        REG_LEN = KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
        LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
        PACKED_SCAN_SIZE = COUNTER_LANES * BLOCK_THREADS * sizeof(CounterT),
        DIGIT_BITS = sizeof(DigitT) << 3,
        DIGIT_MASK = (1 << DIGIT_BITS) - 1,
        KEY_TRAITS_TYPE_MASK = 1l << ((sizeof(KeyTraitsT) << 3) - 1),
    };

private:
    int lid;

public:
    static int get_shared_local_memory_size() {
        constexpr int KV_TYPE_SIZE = KEYS_ONLY ? sizeof(KeyT) : std::max(sizeof(KeyT), sizeof(ValueT));
        return std::max(BLOCK_THREADS * KEYS_PER_THREAD * KV_TYPE_SIZE, (int)PACKED_SCAN_SIZE);
    }

    GPU_CODE inline SegmentedBlockRadixSort(int lid) :
        lid(lid) {
    }

    GPU_CODE inline void sort(
        KeyTraitsT (&pkey)[KEYS_PER_THREAD],
        ValueT (&pvalue)[KEYS_PER_THREAD]) {
        int begin_bit = 0;
        int end_bit = 8 * sizeof(KeyTraitsT);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int rank[KEYS_PER_THREAD];
            rank_keys(pkey, rank, begin_bit, pass_bits);
            exchange<KeyTraitsT>(pkey, rank);
            if (!KEYS_ONLY) exchange<ValueT>(pvalue, rank);
            begin_bit += RADIX_BITS;
            if (begin_bit >= end_bit) break;
        }
    }

    GPU_CODE inline void sort(KeyTraitsT (&pkey)[KEYS_PER_THREAD]) {
        int begin_bit = 0;
        int end_bit = 8 * sizeof(KeyTraitsT);
        while (true) {
            auto pass_bits = NUMERIC_MIN(RADIX_BITS, end_bit - begin_bit);
            int rank[KEYS_PER_THREAD];
            rank_keys(pkey, rank, begin_bit, pass_bits);
            exchange<KeyTraitsT>(pkey, rank);
            begin_bit += RADIX_BITS;
            if (begin_bit >= end_bit) break;
        }
    }

    GPU_CODE inline void read_key_from_global(
        KeyTraitsT (&pkey)[KEYS_PER_THREAD],
        KeyT *key, int length) {
        KeyTraitsT PADDING_KEY;
        if (IS_DESCENDING) {
            PADDING_KEY = 0;
        } else {
            PADDING_KEY = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
            PADDING_KEY = PADDING_KEY ^ (PADDING_KEY - 1);
        }
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid * KEYS_PER_THREAD + ITEM;
            pkey[ITEM] = (offset < length) ? KeyTraits<KeyT>::convert(key[offset]) :
                                             PADDING_KEY;
        }
    }

    GPU_CODE inline void read_value_from_global(
        PrivateValueT (&pvalue)[KEYS_PER_THREAD],
        ValueT *value, int length) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid * KEYS_PER_THREAD + ITEM;
            if (!USE_INDICES_AS_VALUE) {
                if (offset < length)
                    pvalue[ITEM] = value[offset];
            } else {
                pvalue[ITEM] = offset;
            }
        }
    }

    GPU_CODE inline void write_key_to_global(
        KeyT *key,
        KeyTraitsT (&pkey)[KEYS_PER_THREAD], int length) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid * KEYS_PER_THREAD + ITEM;
            if (offset < length)
                key[offset] = KeyTraits<KeyT>::deconvert(pkey[ITEM]);
        }
    }

    GPU_CODE inline void write_value_to_global(
        ValueT *value,
        PrivateValueT (&pvalue)[KEYS_PER_THREAD], int length) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid * KEYS_PER_THREAD + ITEM;
            if (offset < length) value[offset] = pvalue[ITEM];
        }
    }

    template <typename T>
    GPU_CODE inline void exchange(
        T (&data)[KEYS_PER_THREAD],
        int (&rank)[KEYS_PER_THREAD]) {
        auto local_storage_ = reinterpret_cast<T *>(local_storage);
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
            local_storage_[rank[ITEM]] = data[ITEM];
        __syncthreads();
        auto local_storage_lid = local_storage_ + lid * KEYS_PER_THREAD;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
            data[ITEM] = local_storage_lid[ITEM];
        __syncthreads();
    }

    GPU_CODE inline DigitT extract_digit(KeyTraitsT key, int begin, int pass) {
        return ((key >> begin) & ((1 << pass) - 1));
    }

    GPU_CODE inline void rank_keys(
        KeyTraitsT (&key)[KEYS_PER_THREAD],
        int (&rank)[KEYS_PER_THREAD],
        int begin_bit,
        int pass_bits) {
        DigitT *digit_counters[KEYS_PER_THREAD];
        DigitT sub_counters[KEYS_PER_THREAD];
        auto scan_storage = reinterpret_cast<CounterT *>(local_storage);
        auto buckets = reinterpret_cast<DigitT(*)[BLOCK_THREADS][PACKING_RATIO]>(local_storage);

        // Reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM)
            scan_storage[lid * COUNTER_LANES + ITEM] = 0; // fast
        __syncthreads();

        // Bin
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(key[ITEM], begin_bit, pass_bits);
            auto sub_counter = digit >> LOG_COUNTER_LANES;
            auto counter_lane = digit & (COUNTER_LANES - 1);
            if (IS_DESCENDING) {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }
            sub_counters[ITEM] = sub_counter;
            digit_counters[ITEM] = &buckets[counter_lane][lid][sub_counter];
            rank[ITEM] = *digit_counters[ITEM];
            *digit_counters[ITEM] = rank[ITEM] + 1;
        }
        __syncthreads();

        // Exclusive scan
        CounterT temp = block_aligned_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS>(scan_storage, lid);

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
        __syncthreads();
    }
};

} // namespace pmkl
