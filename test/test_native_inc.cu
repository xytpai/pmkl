#include "pmkl.h"
#include <iostream>

using namespace pmkl;
using namespace std;

int main() {
    auto l = GpuLauncher::GetInstance();
    const int prb_size = 1024;
    using dtype = int;
    dtype x_cpu[prb_size];
    for (auto &x : x_cpu) x = 0;
    auto x_gpu = l->malloc<dtype>(prb_size);
    l->stream_begin();
    l->memcpy<dtype>(x_gpu, x_cpu, prb_size, GpuLauncher::Direction::H2D, false);
    l->stream_sync();
    l->submit(
        0, {1}, {prb_size},
        [=] DEVICE_LAMBDA(KernelInfo & info) { x_gpu[info.t0] += info.t0; });
    l->stream_sync();
    l->memcpy<dtype>(x_cpu, x_gpu, prb_size, GpuLauncher::Direction::D2H, false);
    l->stream_end();
    for (auto x : x_cpu) cout << x << endl;
}
