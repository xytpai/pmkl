#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
    {
        Tensor a = empty({3, 1, 3, 1, 1}, ScalarType::Float, 0);
        Tensor b = empty({1, 2, 3, 1, 1}, ScalarType::Float, 0);
        Tensor c = empty({1, 2, 3, 4, 1}, ScalarType::Float, 0);
        Tensor d = empty({3, 2, 3, 4, 5}, ScalarType::Float, 0);
        Tensor e = empty({3, 2, 3, 4, 5}, ScalarType::Float, 0);
        auto iter = TensorIterator()
                        .add_output(a)
                        .add_output(b)
                        .add_input(c)
                        .add_input(d)
                        .add_input(e);

        if (iter.ntensors() != 5) return 1;
        if (iter.ninputs() != 3) return 1;
        if (iter.noutputs() != 2) return 1;

        if (a.storage_ref_count() != 1) return 1;
        if (b.storage_ref_count() != 1) return 1;
        if (c.storage_ref_count() != 1) return 1;
        if (d.storage_ref_count() != 1) return 1;
        if (e.storage_ref_count() != 1) return 1;

        if (iter.tensor(3).shape(4) != 5) return 1;
        if (iter.tensor(3).shape(3) != 4) return 1;

        if (iter.is_contiguous() == true) return 1;

        iter.build_for_loops();
    }

    {
        Tensor left = empty({12, 32, 44}, ScalarType::Float, 0);
        Tensor right = empty({12, 32, 44}, ScalarType::Float, 0);
        Tensor out;
        auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
        if (iter.is_contiguous() == false) return 1;
    }

    {
        Tensor left = empty({12, 1, 44}, ScalarType::Float, 0);
        Tensor right = empty({12, 32, 1}, ScalarType::Float, 0);
        Tensor out;
        auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
        if (out.storage_ref_count() != 1) return 1;
        if (out.shape(0) != 12) return 1;
        if (out.shape(1) != 32) return 1;
        if (out.shape(2) != 44) return 1;
        if (iter.stride_bytes(1, 2) != 0) return 1;
        if (iter.stride_bytes(2, 0) != 0) return 1;
        if (iter.stride_bytes(0, 2) != 1408 * 4) return 1;
        if (iter.is_contiguous() == true) return 1;

        cout << out << endl;
        cout << iter << endl;
    }

    cout << "ok\n";
    return 0;
}
