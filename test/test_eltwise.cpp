#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
    auto l = GpuLauncher::GetInstance();
    {
        Tensor left = empty({10}, ScalarType::Long, 0);
        Tensor right = empty({1}, ScalarType::Short, 0);
        long left_[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        short right_[1] = {2};
        left.copy_from_cpu_ptr((void *)left_);
        right.copy_from_cpu_ptr((void *)right_);

        l->stream_begin();
        auto out = left + right;
        l->stream_sync();
        l->stream_end();

        long out_[10];
        out.copy_to_cpu_ptr((void *)out_);
        for (int i = 0; i < 10; i++) cout << out_[i] << endl;
    }

    cout << "ok\n";
    return 0;
}
