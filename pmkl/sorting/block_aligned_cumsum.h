#pragma once

#include "launcher.h"

namespace pmkl {
namespace sorting {

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
DEVICE inline void warp_aligned_cumsum(const int wid, const T input, T &inclusive_sum, T &exclusive_sum) {
    inclusive_sum = input;
#pragma unroll
    for (int i = 0, offset = 1; i < STEPS; ++i, offset <<= 1) {
#if defined(USE_CUDA)
        T temp = __shfl_up_sync(0xffffffff, inclusive_sum, offset);
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

    int warp_local_id = lid % WARP_SIZE;
    int warp_id = lid / WARP_SIZE;
    int lane_temp_values[COUNTER_LANES];

    // Read input lane sum
    auto storage_lanes = storage + lid * COUNTER_LANES;
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
