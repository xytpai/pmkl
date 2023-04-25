#pragma once

#include "core.h"
#include "key_traits.h"

namespace pmkl {
namespace sorting {

template <typename T>
DEVICE_INLINE T NUMERIC_MIN(T A, T B) {
    return (((A) > (B)) ? (B) : (A));
}

template <typename T>
DEVICE_INLINE T DivUp(T A, T B) {
    return (((A) + (B)-1) / (B));
}

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

/*
The computational pipeline is as following:
warp0               |  warp1
in0  in1  in2  in3  |  in4  in5  in6  in7
1    1    1    1    |  1    1    1    1
-> warp_scan
1    2    3    4    |  1    2    3    4   <-inclusive_sum
0    1    2    3    |  0    1    2    3   <-exclusive_sum
STEPS should be Log2<WARP_SIZE>::VALUE
*/
template <typename T, int STEPS>
DEVICE inline void warp_aligned_cumsum(KernelInfo &info, const int wid, const T input, T &inclusive_sum, T &exclusive_sum) {
    inclusive_sum = input;
#pragma unroll
    for (int i = 0; i < STEPS; ++i) {
        uint32_t offset = 1u << i;
#if defined(USE_CUDA)
        T temp = __shfl_up_sync(0xffffffff, inclusive_sum, offset);
#elif defined(USE_DPCPP)
        T temp = sycl::shift_group_right(info.item_.get_sub_group(), inclusive_sum, offset);
#endif
        if (wid >= offset) inclusive_sum += temp;
    }
    exclusive_sum = inclusive_sum - input;
}

/*
Perform cumsum blockly.
The input sequence shuold have fixed size : prb_size = COUNTER_LANES * THREADS
Note that a thread handle COUNTER_LANES items of contiguous memory.
*/
template <
    typename T,
    int COUNTER_LANES,
    int THREADS,
    bool EXCLUSIVE = true,
    int WARP_SIZE = 32>
DEVICE inline T block_aligned_cumsum(KernelInfo &info, T *storage, const int lid) {
    static_assert(THREADS % WARP_SIZE == 0, "THREADS should be n * WARP_SIZE. (n = 1, 2, 3, ...)");

    const int NUM_WARPS = THREADS / WARP_SIZE;
    const int WARP_CUMSUM_STEPS = Log2<WARP_SIZE>::VALUE;

#if defined(USE_CUDA)
    int warp_local_id = lid % WARP_SIZE;
    int warp_id = lid / WARP_SIZE;
#elif defined(USE_DPCPP)
    auto sg = info.item_.get_sub_group();
    int warp_local_id = sg.get_local_id();
    int warp_id = sg.get_group_linear_id();
#endif
    int lane_temp_values[COUNTER_LANES];

    // Read input lane sum
    auto storage_lanes = &storage[lid * COUNTER_LANES];
    T lane_all_sum = 0;

    if (EXCLUSIVE) {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_temp_values[lane] = lane_all_sum;
            lane_all_sum += storage_lanes[lane];
        }
    } else {
#pragma unroll
        for (int lane = 0; lane < COUNTER_LANES; ++lane) {
            lane_all_sum += storage_lanes[lane];
            lane_temp_values[lane] = lane_all_sum;
        }
    }

    // Get warp level exclusive sum
    T warp_inclusive_sum, warp_exclusive_sum;
    warp_aligned_cumsum<T, WARP_CUMSUM_STEPS>(
        info,
        warp_local_id,
        lane_all_sum,
        warp_inclusive_sum,
        warp_exclusive_sum);
    info.barrier();

    // Write to storage
    if (warp_local_id == (WARP_SIZE - 1))
        storage[warp_id] = warp_inclusive_sum;
    info.barrier();

    // Get block prefix
    T block_all_sum = 0, block_exclusive_sum;
#pragma unroll
    for (int i = 0; i < NUM_WARPS; ++i) {
        if (warp_id == i)
            block_exclusive_sum = block_all_sum;
        block_all_sum += storage[i];
    }
    info.barrier();

    // Write to storage
    warp_exclusive_sum += block_exclusive_sum;
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
        storage_lanes[lane] = warp_exclusive_sum + lane_temp_values[lane];
    }
    info.barrier();

    return block_all_sum;
}

template <typename T, int COUNTER_LANES, int THREADS>
DEVICE inline T block_aligned_exclusive_cumsum(KernelInfo &info, T *slm_storage, const int lid) {
    return block_aligned_cumsum<T, COUNTER_LANES, THREADS, true>(info, slm_storage, lid);
}

template <typename T, int COUNTER_LANES, int THREADS>
DEVICE inline T block_aligned_inclusive_cumsum(KernelInfo &info, T *slm_storage, const int lid) {
    return block_aligned_cumsum<T, COUNTER_LANES, THREADS, false>(info, slm_storage, lid);
}

}
} // namespace pmkl::sorting
