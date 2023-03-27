#include <iostream>
#include <algorithm>

#include "pmkl.h"
#include "warp_softmax.h"

using namespace pmkl;
using namespace std;

template <typename output_t, typename input_t, typename acc_t>
void host_softmax_forward(
    output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count) {
    for (int b = 0; b < batch_count; b++) {
        auto src_b = src + b * softmax_elements_stride;
        auto dst_b = dst + b * softmax_elements_stride;
        acc_t max_v = -99999999;
        for (int i = 0; i < softmax_elements; i++) {
            max_v = src_b[i] > max_v ? src_b[i] : max_v;
        }
        acc_t sum = 0;
        for (int i = 0; i < softmax_elements; i++) {
            sum += std::exp(src_b[i] - max_v);
        }
        for (int i = 0; i < softmax_elements; i++) {
            dst_b[i] = std::exp(src_b[i] - max_v) / sum;
        }
    }
}

int main() {
    auto l = GpuLauncher::GetInstance();
    using scalar_t = float;
    const int batch_size = 12;
    const int eltsize = 1024; // 0.83456 ms 3070
    const int numel = batch_size * eltsize;

    scalar_t *input_cpu = new scalar_t[numel];
    utils::host::fill_rand<scalar_t>(input_cpu, numel, -10.0, 10.0);
    scalar_t *input = l->malloc<scalar_t>(numel);
    l->memcpy((void *)input, (void *)input_cpu, numel * sizeof(scalar_t), GpuLauncher::Direction::H2D);

    scalar_t *output = l->malloc<scalar_t>(numel);
    scalar_t *output_cpu = new scalar_t[numel];
    scalar_t *outout_gpu_cpu = new scalar_t[numel];

    l->stream_begin();
    l->set_profiling_mode(true);
    for (int i = 0; i < 10; i++)
        norm::warp_softmax_forward<scalar_t, scalar_t, scalar_t, false>(output, input, eltsize, eltsize, batch_size);
    l->stream_end();
    l->memcpy((void *)outout_gpu_cpu, (void *)output, numel * sizeof(scalar_t), GpuLauncher::Direction::D2H);

    host_softmax_forward<scalar_t, scalar_t, scalar_t>(output_cpu, input_cpu, eltsize, eltsize, batch_size);

    if (!utils::all_close(outout_gpu_cpu, output_cpu, numel))
        return 1;
    cout << "ok";

    return 0;
}

/*
xor
0.014336 ms
0.008192 ms
0.007168 ms
0.007168 ms
0.007168 ms
0.006912 ms
0.007104 ms
0.00784 ms
0.007168 ms
0.007008 ms

down
0.015104 ms
0.008768 ms
0.008032 ms
0.008 ms
0.007168 ms
0.007168 ms
0.007168 ms
0.007168 ms
0.008128 ms
0.007168 ms
*/
