#pragma once

#include <limits>
#include <tuple>
#include <utility>

#include "function_traits.h"
#include "tensor_iterator.h"
#include "tensor_offset_calculator.h"
#include "tensor_memory_access.h"
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

template <class F, class Tuple>
HOST_DEVICE_INLINE constexpr decltype(auto) pmkl_loops_apply(F &&f, Tuple &&t) {
    return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

template <int THREAD_WORK_SIZE, typename func_t, typename policy_t>
DEVICE_INLINE void elementwise_kernel_helper(func_t f, policy_t policy) {
    using traits = function_traits<func_t>;
    using return_t = typename traits::result_type;
    using args_t = typename traits::ArgsTuple;
    return_t results[THREAD_WORK_SIZE];
    args_t args[THREAD_WORK_SIZE];
    policy.load(args);
#pragma unroll
    for (int i = 0; i < THREAD_WORK_SIZE; i++) {
        if (policy.check_inbounds(i)) {
            results[i] = pmkl_loops_apply(f, args[i]);
        }
    }
    policy.store(results);
}

// template<typename func_t, typename array_t, typename inp_calc_t>
// static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data, inp_calc_t input_calc, int vec_size) {
//   CHECK_FAIL(N > 0 && N <= std::numeric_limits<int32_t>::max());
//   using traits = function_traits<func_t>;
//   int64_t grid = (N + block_work_size() - 1) / block_work_size();

//   auto stream = at::cuda::getCurrentCUDAStream();
//   int vec_size = memory::can_vectorize_up_to<func_t>(data);

//   switch (vec_size) {
//   case 4:
//     vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   case 2:
//     vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   case 1: {
//     auto input_calc = TrivialOffsetCalculator<traits::arity>();
//     auto output_calc = TrivialOffsetCalculator<1>();
//     auto loader = memory::LoadWithoutCast();
//     auto storer = memory::StoreWithoutCast();
//     unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   }
//   default:
//     TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
//   }
// }

// template<typename func_t, typename array_t>
// static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
//   CHECK_FAIL(N > 0 && N <= std::numeric_limits<int32_t>::max());
//   using traits = function_traits<func_t>;
//   int64_t grid = (N + block_work_size() - 1) / block_work_size();
//   auto stream = at::cuda::getCurrentCUDAStream();
//   int vec_size = memory::can_vectorize_up_to<func_t>(data);

//   switch (vec_size) {
//   case 4:
//     vectorized_elementwise_kernel<4, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   case 2:
//     vectorized_elementwise_kernel<2, func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   case 1: {
//     auto input_calc = TrivialOffsetCalculator<traits::arity>();
//     auto output_calc = TrivialOffsetCalculator<1>();
//     auto loader = memory::LoadWithoutCast();
//     auto storer = memory::StoreWithoutCast();
//     unrolled_elementwise_kernel<func_t, array_t><<<grid, num_threads(), 0, stream>>>(N, f, data, input_calc, output_calc, loader, storer);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     break;
//   }
//   default:
//     TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
//   }
// }

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
