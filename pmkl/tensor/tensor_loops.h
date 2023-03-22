#pragma once

#include "function_traits.h"
#include "tensor_iterator.h"
#include "tensor_offset.h"
#include "exception.h"
#include "array.h"

namespace pmkl {

template <typename func_t, int nargs = function_traits<func_t>::arity>
struct needs_dynamic_casting {
    static bool check(TensorIterator &iter) {
        using traits = function_traits<func_t>;
        using cpp_type = typename traits::template arg<nargs - 1>::type;
        using cpp_map = CppTypeToScalarType<cpp_type>;

        if (iter.input_dtype(nargs - 1) != cpp_map::value) {
            return true;
        }
        return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
    }
};

template <typename func_t>
struct needs_dynamic_casting<func_t, 0> {
    static bool check(TensorIterator &iter) {
        using traits = function_traits<func_t>;
        using cpp_type = typename traits::result_type;
        if constexpr (std::is_void<cpp_type>::value) {
            return false;
        } else {
            return iter.dtype(0) != CppTypeToScalarType<cpp_type>::value;
        }
        return true;
    }
};

template <typename func_t>
void gpu_kernel_impl(TensorIterator &iter, const func_t &f) {
    using traits = function_traits<func_t>;
    using arg0_t = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;

    CHECK_FAIL(iter.can_use_32bit_indexing());
    CHECK_FAIL(iter.ninputs() == traits::arity);
    CHECK_FAIL(iter.noutputs() == 1);

    memory::array<char *, ntensors> data;
    for (int i = 0; i < ntensors; i++) {
        data[i] = (char *)iter.data_ptr(i);
    }

    int64_t numel = iter.numel();

    bool contiguous = iter.is_contiguous();
    bool dynamic_casting = needs_dynamic_casting<func_t>::check(iter);

    if (!dynamic_casting) {
        // if (contiguous) {
        //     launch_vectorized_kernel(numel, f, data);
        // } else {
        //     auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        //     constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
        //     launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
        //         auto offsets = offset_calc.get(idx);
        //         arg0_t *out = (arg0_t *)(data[0] + offsets[0]);
        //         *out = invoke(f, &data.data[1], &offsets.data[1], 1);
        //     });
        // }
    } else {
        //     if (contiguous) {
        //         at::detail::Array<ScalarType, traits::arity> dtypes;
        //         for (int i = 0; i < traits::arity; i++) {
        //             dtypes[i] = iter.dtype(i + 1);
        //         }
        //         auto loader = memory::LoadWithCast<traits::arity>(dtypes);
        //         auto storer = memory::StoreWithCast(iter.dtype(0));
        //         auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
        //         auto output_offset_calculator = TrivialOffsetCalculator<1>();
        //         launch_unrolled_kernel(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
        //     } else {
        //         at::detail::Array<ScalarType, ntensors> dtypes;
        //         for (int i = 0; i < ntensors; i++) {
        //             dtypes[i] = iter.dtype(i);
        //         }
        //         auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
        //         launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
        //             auto offsets = offset_calc.get(idx);
        //             void *out = data[0] + offsets[0];
        //             arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
        //             c10::cast_and_store<arg0_t>(dtypes[0], out, result);
        //         });
        //     }
    }
}

} // namespace pmkl
