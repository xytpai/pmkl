#include "pmkl.h"
#include <iostream>

using namespace std;
using namespace pmkl;

int main() {
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

    if (a.storage_ref_count() != 2) return 1;
    if (b.storage_ref_count() != 2) return 1;
    if (c.storage_ref_count() != 2) return 1;
    if (d.storage_ref_count() != 2) return 1;
    if (e.storage_ref_count() != 2) return 1;

    if (iter.tensor(3).shape(4) != 5) return 1;
    if (iter.tensor(3).shape(3) != 4) return 1;

    iter.build_for_loops();

    // Tensor oo = empty({12, 32, 44}, ScalarType::Float, 0);
    Tensor oo;
    Tensor aa = empty({12, 32, 1}, ScalarType::Float, 0);
    Tensor bb = empty({12, 1, 44}, ScalarType::Float, 0);
    auto iter2 = TensorIterator().add_output(oo).add_input(aa).add_input(bb).build_for_loops();

    std::cout << iter2 << std::endl;
    std::cout << iter2.outputs(0);

    cout << "ok\n";
    return 0;
}
