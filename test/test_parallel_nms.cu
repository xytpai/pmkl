#include "pmkl.h"
#include "parallel_nms.h"
#include <iostream>

using namespace std;
using namespace pmkl;
using namespace pmkl::utils;

void gen_bitmap(unsigned long long *dst, int batch_size, int n_elements) {
    // ->dst: batch_size * n_elements * n_chunks
    constexpr int CHUNK_SIZE = sizeof(unsigned long long) * 8;
    int n_chunks = (n_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (int i = 0; i < batch_size * n_elements * n_chunks; i++) dst[i] = 0ULL;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_elements; i++) {
            for (int j = i + 1; j < n_elements; j++) {
                int div = j / CHUNK_SIZE;
                int mod = j % CHUNK_SIZE;
                if (utils::host::randint_scalar(0, 2) > 0) {
                    dst[(b * n_elements + i) * n_chunks + div] |= 1ULL << mod;
                }
            }
        }
    }
}

void parallel_nms_cpu(
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

    for (int b = 0; b < batch_size; b++) {
        auto keeps_ = new bool[n_elements];
        for (int k = 0; k < n_elements; k++) keeps_[k] = true;

        n_keeps[b] = 0;
        for (int i = 0; i < n_elements; i++) {
            if (!keeps_[i]) continue;
            keep_indices[b * n_elements + n_keeps[b]] = i;
            if (++n_keeps[b] >= max_elements) break;
            for (int j = i + 1; j < n_elements; j++) {
                auto vec = intersection_bit_matrix[b * n_elements * n_chunks + i * n_chunks + j / CHUNK_SIZE];
                if ((vec >> (j % CHUNK_SIZE)) & 1)
                    keeps_[j] = false;
            }
        }

        delete[] keeps_;
    }
}

int main() {
    constexpr int CHUNK_SIZE = sizeof(unsigned long long) * 8;
    auto l = GpuLauncher::GetInstance();

    for (int it = 0; it < 40; it++) {
        int batch_size = utils::host::randint_scalar(1, 22);
        ;
        int n_elements = utils::host::randint_scalar(12, 2048);
        int n_chunks = (n_elements + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int max_elements = utils::host::randint_scalar((int)(n_elements * 0.5), n_elements);
        if (it == 0) max_elements = n_elements + 1;

        cout << "batch_size: " << batch_size << ", n_elements:" << n_elements << ", max_elements: " << max_elements << endl;

        unsigned long long *bitmap_cpu = new unsigned long long[batch_size * n_elements * n_chunks];
        int *keep_indices_cpu = new int[batch_size * n_elements];
        int *n_keeps_cpu = new int[batch_size];

        gen_bitmap(bitmap_cpu, batch_size, n_elements);
        parallel_nms_cpu(bitmap_cpu, batch_size, n_elements, max_elements, keep_indices_cpu, n_keeps_cpu);

        auto bitmap_gpu = l->malloc<unsigned long long>(batch_size * n_elements * n_chunks);
        auto keep_indices_gpu = l->malloc<int>(batch_size * n_elements);
        auto n_keeps_gpu = l->malloc<int>(batch_size);

        int *keep_indices_out = new int[batch_size * n_elements];
        int *n_keeps_out = new int[batch_size];

        l->memcpy((void *)bitmap_gpu, (void *)bitmap_cpu,
                  batch_size * n_elements * n_chunks * sizeof(unsigned long long), GpuLauncher::Direction::H2D);
        l->stream_begin();
        image::nms::parallel_nms(bitmap_gpu, batch_size, n_elements, max_elements, keep_indices_gpu, n_keeps_gpu);
        l->stream_sync();
        l->stream_end();

        l->memcpy((void *)keep_indices_out, (void *)keep_indices_gpu, batch_size * n_elements * sizeof(int), GpuLauncher::Direction::D2H);
        l->memcpy((void *)n_keeps_out, (void *)n_keeps_gpu, batch_size * sizeof(int), GpuLauncher::Direction::D2H);

        cout << "testing n_keeps...\n";
        if (!all_close(n_keeps_out, n_keeps_cpu, batch_size))
            return 1;

        cout << "testing keep_indices...\n";
        for (int b = 0; b < batch_size; b++) {
            if (!all_close(keep_indices_out + b * n_elements,
                           keep_indices_cpu + b * n_elements, n_keeps_out[b]))
                return 1;
        }

        delete[] bitmap_cpu;
        delete[] keep_indices_cpu;
        delete[] n_keeps_cpu;
        delete[] keep_indices_out;
        delete[] n_keeps_out;

        l->free(bitmap_gpu);
        l->free(keep_indices_gpu);
        l->free(n_keeps_gpu);
    }

    cout << "ok\n";
    return 0;
}
