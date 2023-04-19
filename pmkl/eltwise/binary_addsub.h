#pragma once

#include "core.h"

namespace pmkl {

Tensor add(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    PMKL_DISPATCH_BASIC_TYPES(out.dtype(), "sub", [&]() {
        gpu_kernel(iter, [] HOST_DEVICE(scalar_t a, scalar_t b) -> scalar_t { return a + b; });
    });
    return out;
}

Tensor sub(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    PMKL_DISPATCH_BASIC_TYPES(out.dtype(), "sub", [&]() {
        gpu_kernel(iter, [] HOST_DEVICE(scalar_t a, scalar_t b) -> scalar_t { return a - b; });
    });
    return out;
}

Tensor operator+(Tensor &left, Tensor &right) {
    return add(left, right);
}

Tensor operator-(Tensor &left, Tensor &right) {
    return sub(left, right);
}

} // namespace pmkl
