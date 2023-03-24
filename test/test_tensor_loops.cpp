#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
    auto l = GpuLauncher::GetInstance();
    {
        Tensor left = empty({10}, ScalarType::Long, 0);
        Tensor right = empty({10}, ScalarType::Long, 0);
        long left_[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        long right_[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        left.copy_from_cpu_ptr((void *)left_);
        right.copy_from_cpu_ptr((void *)right_);

        Tensor out;
        auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
        cout << out.numel() << endl;
        // if (iter.is_contiguous() == false) return 1;
        using scalar_t = long;
        l->stream_begin();
        gpu_kernel(iter, [] HOST_DEVICE(scalar_t a, scalar_t b) -> scalar_t { return a + b; });
        l->stream_sync();
        l->stream_end();

        long out_[10];
        out.copy_to_cpu_ptr((void *)out_);
        for (int i = 0; i < 10; i++) cout << out_[i] << endl;
    }

    cout << "ok\n";
    return 0;
}
