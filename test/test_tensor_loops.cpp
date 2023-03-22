#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
    {
        Tensor left = empty({12, 32, 44}, ScalarType::Float, 0);
        Tensor right = empty({12, 32, 44}, ScalarType::Double, 0);
        Tensor out;
        auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
        // if (iter.is_contiguous() == false) return 1;
        using scalar_t = float;
        gpu_kernel_impl(iter, [](scalar_t a, scalar_t b) -> scalar_t { return a + b; });
    }

    cout << "ok\n";
    return 0;
}
