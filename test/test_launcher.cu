#include "pmkl.h"
#include <iostream>

using namespace pmkl;
using namespace pmkl::utils;
using namespace std;

const int prb_size = 12345;

int main() {
    auto l = GpuLauncher::GetInstance();

    // set device
    cout << "testing device set\n";
    int count = l->device_count();
    cout << "device count: " << count << endl;
    if (l->device() != 0) return 1;
    for (int i = 0; i < count; i++) {
        l->set_device(i);
        if (l->device() != i) return 1;
    }
    l->set_device(0);
    l->reset_device();

    // memcpy
    cout << "testing memcpy\n";
    auto cm = new float[prb_size];
    auto cm2 = new float[prb_size];
    auto gm = l->malloc<float>(prb_size);
    auto gm2 = l->malloc<float>(prb_size);
    for (int i = 0; i < prb_size; i++) cm[i] = 1.3 * i;
    l->memcpy(gm, cm, prb_size, GpuLauncher::Direction::H2D);
    l->memcpy(gm2, gm, prb_size, GpuLauncher::Direction::D2D);
    l->memcpy(cm2, gm2, prb_size, GpuLauncher::Direction::D2H);
    auto ret = all_close<float>(cm2, cm, prb_size);
    if (!ret) return 1;
    delete cm;
    delete cm2;
    l->free(gm);
    l->free(gm2);

    // memset zero
    cout << "testing memset zero\n";
    auto data_gpu = l->malloc<int>(prb_size);
    l->memset((void *)data_gpu, 0, prb_size * sizeof(int));
    auto data_cpu = new int[prb_size];
    for (int i = 0; i < prb_size; i++) data_cpu[i] = i;
    l->memcpy<int>(data_cpu, data_gpu, prb_size, GpuLauncher::Direction::D2H);
    for (int i = 0; i < prb_size; i++) {
        if (data_cpu[i] != 0) return 1;
    }
    l->free(data_gpu);
    delete data_cpu;

    // kernel submit
    cout << "testing kernel submit\n";
    auto x_out = new int[prb_size];
    auto x_gpu = l->malloc<int>(prb_size);
    l->stream_begin();
    int group_size = (prb_size + 256 - 1) / 256;
    l->submit(
        0, {group_size}, {256},
        [=] DEVICE_LAMBDA(KernelInfo & info) {
            int lid = info.b0 * info.t_size0 + info.t0;
            if (lid < prb_size) x_gpu[lid] = 2 * lid;
        });
    l->stream_sync();
    l->submit(
        0, {group_size}, {256},
        [=] DEVICE_LAMBDA(KernelInfo & info) {
            int lid = info.b0 * info.t_size0 + info.t0;
            if (lid < prb_size) x_gpu[lid] -= 1;
        });
    l->stream_sync();
    l->memcpy(x_out, x_gpu, prb_size, GpuLauncher::Direction::D2H, true);
    l->stream_end();
    for (int i = 0; i < prb_size; i++) {
        if (x_out[i] != 2 * i - 1) return 1;
    }
    l->free(x_gpu);
    delete x_out;

    // device info test
    cout << "testing device info\n";
    auto mt = l->max_thread_per_group();
    cout << "max_thread_per_group: " << mt << endl;
    if (mt >= 8192 || mt <= 32) return 1;
    auto slm_size = l->shared_local_memory_size();
    cout << "shared_local_memory_size: " << slm_size << endl;
    if (slm_size <= 32) return 1;

    // kernel submit2
    {
        cout << "testing kernel submit2\n";
        auto x_out = new int[prb_size];
        auto x_gpu = l->malloc<int>(prb_size);
        l->stream_begin();
        int group_size = (prb_size + 256 - 1) / 256;
        l->submit(
            0, {1, group_size}, {16, 8, 2},
            [=] DEVICE_LAMBDA(KernelInfo & info) {
                int t0 = info.t0 * 16 + info.t1 * 2 + info.t2;
                int lid = info.b1 * 256 + t0;
                if (lid < prb_size) x_gpu[lid] = 2 * lid;
            });
        l->stream_sync();
        l->submit(
            0, {1, group_size}, {16, 8, 2},
            [=] DEVICE_LAMBDA(KernelInfo & info) {
                int t0 = info.t0 * 16 + info.t1 * 2 + info.t2;
                int lid = info.b1 * 256 + t0;
                if (lid < prb_size) x_gpu[lid] -= 1;
            });
        l->stream_sync();
        l->memcpy(x_out, x_gpu, prb_size, GpuLauncher::Direction::D2H, true);
        l->stream_end();
        for (int i = 0; i < prb_size; i++) {
            if (x_out[i] != 2 * i - 1) {
                cout << "failed\n";
                return 1;
            }
        }
        l->free(x_gpu);
        delete x_out;
    }

    return 0;
}
