#pragma once

#include <limits>
#include <algorithm>

#include "core.h"

namespace pmkl {
namespace norm {

template <typename T>
struct Add {
    DEVICE_INLINE T operator()(T a, T b) const {
        return a + b;
    }
};

template <typename T>
struct Max {
    DEVICE_INLINE T operator()(T a, T b) const {
        return a < b ? b : a;
    }
};

inline int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

// template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp>
// DEVICE_INLINE void warp_reduce(acc_t *sum) {
//     ReduceOp<acc_t> r;
// #pragma unroll
//     for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
// #pragma unroll
//         for (int i = 0; i < WARP_BATCH; ++i) {
//             acc_t b = GPU_SHFL_XOR(sum[i], offset, WARP_SIZE);
//             sum[i] = r(sum[i], b);
//         }
//     }
// }

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template <typename> class ReduceOp, typename info_t>
DEVICE_INLINE void warp_reduce(info_t &info, acc_t *sum) {
    ReduceOp<acc_t> r;
    int warp_local_idx = threadIdx.x;
    int warp_idx = threadIdx.y;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = GPU_SHFL_DOWN(info, sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
    __shared__ acc_t datax[4 * WARP_BATCH];
    if (warp_local_idx == 0) {
#pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            datax[warp_idx * WARP_BATCH + i] = sum[i];
        }
    }
    __syncthreads();
    if (warp_local_idx != 0) {
#pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            sum[i] = datax[warp_idx * WARP_BATCH + i];
        }
    }
    __syncthreads();
}

template <typename output_t, typename input_t, typename acc_t, int log2_elements, bool is_log_softmax, typename info_t>
DEVICE void warp_softmax_forward_impl(info_t &info, output_t *dst, const input_t *src, int batch_size, int stride, int element_count) {
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < GPU_WARP_SIZE) ? next_power_of_two : GPU_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (info.thread_range(1) * info.block_idx(0) + info.thread_idx(1)) * WARP_BATCH;
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    int local_idx = info.thread_idx(0);
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i * element_count + it * WARP_SIZE];
            } else {
                elements[i][it] = -std::numeric_limits<acc_t>::infinity();
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        max_value[i] = elements[i][0];
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(info, max_value);

    acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            if (is_log_softmax) {
                sum[i] += std::exp(elements[i][it] - max_value[i]);
            } else {
                elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(info, sum);

    // store result
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
#pragma unroll
        for (int it = 0; it < WARP_ITERATIONS; ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i * element_count + it * WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i * element_count + it * WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
                } else {
                    dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}

template <typename output_t, typename input_t, typename acc_t, bool is_log_softmax>
void warp_softmax_forward(
    output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count) {
    CHECK_FAIL(softmax_elements >= 0 && softmax_elements <= 1024);
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;
        int warp_size = GPU_WARP_SIZE;
        warp_size = (next_power_of_two < warp_size) ? next_power_of_two : warp_size;
        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        auto l = GpuLauncher::GetInstance();
#define LAUNCH_WARP_SOFTMAX_FORWARD(L2E)                                                                                                                                                          \
    case L2E:                                                                                                                                                                                     \
        l->submit(                                                                                                                                                                                \
            4 * batches_per_warp,                                                                                                                                                                 \
            {(batch_count + batches_per_block - 1) / batches_per_block}, {warp_size, warps_per_block, 1},                                                                                         \
            [=] DEVICE(KernelInfo & info) { warp_softmax_forward_impl<output_t, input_t, acc_t, L2E, is_log_softmax>(                                                                             \
                                                info, dst, src, batch_count, softmax_elements_stride, softmax_elements); }); \
        break;
        switch (log2_elements) {
            LAUNCH_WARP_SOFTMAX_FORWARD(0);  // 1
            LAUNCH_WARP_SOFTMAX_FORWARD(1);  // 2
            LAUNCH_WARP_SOFTMAX_FORWARD(2);  // 4
            LAUNCH_WARP_SOFTMAX_FORWARD(3);  // 8
            LAUNCH_WARP_SOFTMAX_FORWARD(4);  // 16
            LAUNCH_WARP_SOFTMAX_FORWARD(5);  // 32
            LAUNCH_WARP_SOFTMAX_FORWARD(6);  // 64
            LAUNCH_WARP_SOFTMAX_FORWARD(7);  // 128
            LAUNCH_WARP_SOFTMAX_FORWARD(8);  // 256
            LAUNCH_WARP_SOFTMAX_FORWARD(9);  // 512
            LAUNCH_WARP_SOFTMAX_FORWARD(10); // 1024
        default:;
        }
#undef LAUNCH_WARP_SOFTMAX_FORWARD
    }
}
}
} // namespace pmkl::norm
