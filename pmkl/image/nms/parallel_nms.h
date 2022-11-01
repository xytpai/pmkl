#pragma once

#include "launcher.h"
#include "exception.h"

namespace pmkl {
namespace image {
namespace nms {

void parallel_nms(
    const unsigned long long *intersection_bit_matrix,
    const int batch_size,
    const int n_elements,
    const int max_elements,
    int *keep_indices,
    int *n_keeps) {
    // intersection_bit_matrix: batch_size * n_elements * CEIL(n_elements, sizeof(unsigned long long) * 8)
    // -> keep_indices: batch_size * n_elements
    // -> n_keeps: batch_size
    constexpr int CHUNK_SIZE = sizeof(unsigned long long) * 8;
    int n_chunks = (n_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;

    auto l = GpuLauncher::GetInstance();
    int threads_per_block = l->max_thread_per_block();

    CHECK_FAIL(n_elements <= threads_per_block * CHUNK_SIZE); // should <= 65536

    l->submit(
        threads_per_block * sizeof(unsigned long long),
        {batch_size}, {threads_per_block},
        [=] DEVICE(KernelInfo & info) {
            int b = info.block_idx(0);
            int lid = info.thread_idx(0);
            auto inputs = intersection_bit_matrix + b * n_elements * n_chunks;
            auto keep_indices_b = keep_indices + b * n_elements;

            auto selected_mask = reinterpret_cast<unsigned long long *>(info.shared_ptr());
            selected_mask[lid] = 0xffffffffffffffffULL;
            info.barrier();

            int n_keeps_ = 0;
            for (int i = 0; i < n_elements; ++i) {
                int div = i / threads_per_block;
                int mod = i % threads_per_block;
                if (!((selected_mask[mod] >> div) & 1ULL)) continue;

                if (lid == 0) keep_indices_b[n_keeps_] = i;
                if (++n_keeps_ >= max_elements) break;

                for (int j = lid + i + 1; j < n_elements; j += threads_per_block) {
                    int div_ = j / threads_per_block;
                    int mod_ = j % threads_per_block;
                    auto vec = inputs[i * n_chunks + j / CHUNK_SIZE];
                    if ((vec >> (j % CHUNK_SIZE)) & 1)
                        selected_mask[mod_] &= ~(1ULL << div_);
                }
                info.barrier();
            }

            if (lid == 0) n_keeps[b] = n_keeps_;
        });
    if (l->is_sync_mode()) l->stream_sync();
}

}
}
} // namespace pmkl::image::nms