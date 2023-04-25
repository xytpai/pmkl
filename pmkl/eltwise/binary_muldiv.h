#pragma once

#include "core.h"

namespace pmkl {

Tensor mul(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    PMKL_DISPATCH_BASIC_TYPES(out.dtype(), "mul", [&]() {
        gpu_kernel(iter, [] HOST_DEVICE(scalar_t a, scalar_t b) -> scalar_t { return a * b; });
    });
    return out;
}

Tensor div(Tensor &left, Tensor &right) {
    Tensor out;
    auto iter = TensorIterator().add_output(out).add_input(left).add_input(right).build_for_loops();
    PMKL_DISPATCH_BASIC_TYPES(out.dtype(), "div", [&]() {
        gpu_kernel(iter, [] HOST_DEVICE(scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
    return out;
}

Tensor operator*(Tensor &left, Tensor &right) {
    return mul(left, right);
}

Tensor operator/(Tensor &left, Tensor &right) {
    return div(left, right);
}

} // namespace pmkl
