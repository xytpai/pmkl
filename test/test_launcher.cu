#include "pmkl.h"
#include <iostream>

using namespace pmkl;
using namespace pmkl::utils;
using namespace std;

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

    // memset zero
    cout << "testing memset zero\n";
    const int prb_size = 12345;
    auto data_gpu = l->malloc<int>(prb_size);
    l->memset((void *)data_gpu, 0, prb_size * sizeof(int));
    auto data_cpu = new int[prb_size];
    for (int i = 0; i < prb_size; i++) data_cpu[i] = i;
    l->memcpy<int>(data_cpu, data_gpu, prb_size, GpuLauncher::Direction::D2H);
    for (int i = 0; i < prb_size; i++) {
        if (data_cpu[i] != 0) return 1;
    }

    return 0;
}
