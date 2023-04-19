#pragma once

#include <limits>
#include <tuple>
#include <utility>

#include "function_traits.h"
#include "tensor_iterator.h"
#include "scalar_type.h"
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

template <int THREAD_WORK_SIZE, int vec_size, typename func_t, typename array_t, typename inp_calc_t>
void vectorized_elementwise_kernel(int N, func_t f, array_t data, inp_calc_t inp_calc) {
    using traits = function_traits<func_t>;
    auto l = GpuLauncher::GetInstance();
    int block_size = l->work_size_for_loops();
    int block_work_size = vec_size * block_size;
    l->submit(
        0, {(N + block_work_size - 1) / block_work_size}, {block_size},
        [=] DEVICE(KernelInfo & info) {
            int remaining = N - block_work_size * info.block_idx(0);
            if (remaining < block_work_size) {
                auto output_calc = TrivialOffsetCalculator<1>();
                auto loader = memory_access::LoadWithoutCast();
                auto storer = memory_access::StoreWithoutCast();
                auto policy = memory_access::policies::unroll<THREAD_WORK_SIZE, array_t,
                                                              decltype(inp_calc), decltype(output_calc),
                                                              memory_access::LoadWithoutCast, memory_access::StoreWithoutCast>(
                    data, remaining, inp_calc, output_calc, loader, storer,
                    info.thread_idx(0), info.block_idx(0), info.thread_range(0));
                elementwise_kernel_helper<THREAD_WORK_SIZE>(f, policy);
            } else {
                elementwise_kernel_helper<THREAD_WORK_SIZE>(f, memory_access::policies::vectorized<
                                                                   THREAD_WORK_SIZE, vec_size, array_t, inp_calc_t>(
                                                                   data, inp_calc, info.thread_idx(0), info.block_idx(0), info.thread_range(0)));
            }
        });
}

template <int THREAD_WORK_SIZE, typename func_t, typename array_t, typename inp_calc_t, typename out_calc_t, typename loader_t, typename storer_t>
static inline void launch_unrolled_kernel(int N, const func_t &f, array_t data,
                                          inp_calc_t ic, out_calc_t oc, loader_t ld, storer_t st) {
    CHECK_FAIL(N > 0 && N <= std::numeric_limits<int32_t>::max());
    auto l = GpuLauncher::GetInstance();
    int block_size = l->work_size_for_loops();
    int block_work_size = THREAD_WORK_SIZE * block_size;
    l->submit(
        0, {(N + block_work_size - 1) / block_work_size}, {block_size},
        [=] DEVICE(KernelInfo & info) {
            int remaining = N - block_work_size * info.block_idx(0);
            auto policy = memory_access::policies::unroll<THREAD_WORK_SIZE, array_t,
                                                          inp_calc_t, out_calc_t, loader_t, storer_t>(data, remaining, ic, oc, ld, st,
                                                                                                      info.thread_idx(0), info.block_idx(0), info.thread_range(0));
            elementwise_kernel_helper<THREAD_WORK_SIZE>(f, policy);
        });
}

template <typename func_t, typename array_t, typename inp_calc_t>
static inline void launch_vectorized_kernel(int N, const func_t &f, array_t data, inp_calc_t input_calc, int vec_size) {
    CHECK_FAIL(N > 0 && N <= std::numeric_limits<int32_t>::max());
    switch (vec_size) {
    case 8:
        vectorized_elementwise_kernel<8, 8, func_t, array_t, inp_calc_t>(N, f, data, input_calc);
        break;
    case 4:
        vectorized_elementwise_kernel<4, 4, func_t, array_t, inp_calc_t>(N, f, data, input_calc);
        break;
    case 2:
        vectorized_elementwise_kernel<2, 2, func_t, array_t, inp_calc_t>(N, f, data, input_calc);
        break;
    case 1: {
        auto output_calc = TrivialOffsetCalculator<1>();
        auto loader = memory_access::LoadWithoutCast();
        auto storer = memory_access::StoreWithoutCast();
        launch_unrolled_kernel<4, func_t, array_t, inp_calc_t>(
            N, f, data, input_calc, output_calc, loader, storer);
        break;
    }
    default:;
    }
}

template <int vec_size, typename func_t>
static void launch_legacy_kernel(int N, const func_t &f) {
    CHECK_FAIL(N >= 0 && N <= std::numeric_limits<int32_t>::max());
    if (N == 0) {
        return;
    }
    auto l = GpuLauncher::GetInstance();
    int block_size = l->work_size_for_loops();
    int block_work_size = vec_size * block_size;
    l->submit(
        0, {(N + block_work_size - 1) / block_work_size}, {block_size},
        [=] DEVICE(KernelInfo & info) {
            int tid = info.thread_idx(0);
            int idx = block_work_size * info.block_idx(0) + tid;
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
                if (idx < N) {
                    f(idx);
                    idx += block_size;
                }
            }
        });
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const data[], const index_t strides[], int i,
            std::index_sequence<INDEX...>) {
    (void)strides;
    (void)i;
    return f(*(typename traits::template arg<INDEX>::type *)(data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const data[], const index_t strides[], int i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
HOST_DEVICE typename traits::result_type
invoke_impl(const func_t &f, char *const data[], const index_t strides[], const ScalarType dtypes[], int i,
            std::index_sequence<I...>) {
    (void)strides;
    (void)i;
    return f(fetch_and_cast<typename traits::template arg<I>::type>(dtypes[I], data[I] + i * strides[I])...);
}

template <typename func_t, typename index_t, typename traits = function_traits<func_t>>
HOST_DEVICE typename traits::result_type
invoke(const func_t &f, char *const data[], const index_t strides[], const ScalarType dtypes[], int i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <typename func_t, typename data_t>
static inline int can_bc_vectorize_up_to(TensorIterator &iter, const data_t &data) {
    if (!iter.has_contiguous_first_dim()) return 1;
    int last_dim_size = iter.shape(0);
    int vec_size = memory_access::can_vectorize_up_to<func_t>(data);
    while (last_dim_size % vec_size) vec_size >>= 1;
    for (int i = 0; i < iter.ntensors(); i++) {
        auto strides = iter.strides(i);
        for (int d = 0; d < iter.ndim(); d++) {
            while (strides[d] % (strides[0] * vec_size)) vec_size >>= 1;
        }
    }
    return vec_size;
}

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
        if (contiguous) {
            int vec_size = memory_access::can_vectorize_up_to<func_t>(data);
            auto input_calc = TrivialOffsetCalculator<traits::arity>();
            launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
        } else {
            int vec_size = can_bc_vectorize_up_to<func_t>(iter, data);
            if (vec_size > 1) {
                auto input_calc = make_input_offset_calculator<traits::arity>(iter);
                launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
            } else {
                auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
                constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
                launch_legacy_kernel<unroll_factor>(numel, [=] HOST_DEVICE(int idx) {
                    auto offsets = offset_calc.get(idx);
                    arg0_t *out = (arg0_t *)(data[0] + offsets[0]);
                    *out = invoke(f, &data.val[1], &offsets.val[1], 1);
                });
            }
        }
    } else {
        if (contiguous) {
            memory::array<ScalarType, traits::arity> dtypes;
            for (int i = 0; i < traits::arity; i++) {
                dtypes[i] = iter.dtype(i + 1);
            }
            auto loader = memory_access::LoadWithCast<traits::arity>(dtypes);
            auto storer = memory_access::StoreWithCast(iter.dtype(0));
            auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
            auto output_offset_calculator = TrivialOffsetCalculator<1>();
            launch_unrolled_kernel<4>(numel, f, data, input_offset_calculator, output_offset_calculator, loader, storer);
        } else {
            memory::array<ScalarType, ntensors> dtypes;
            for (int i = 0; i < ntensors; i++) {
                dtypes[i] = iter.dtype(i);
            }
            auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
            launch_legacy_kernel<4>(numel, [=] HOST_DEVICE(int idx) {
                auto offsets = offset_calc.get(idx);
                void *out = data[0] + offsets[0];
                arg0_t result = invoke(f, &data.val[1], &offsets.val[1], &dtypes.val[1], 1);
                cast_and_store<arg0_t>(dtypes[0], out, result);
            });
        }
    }
}

template <typename func_t>
void gpu_kernel(TensorIterator &iter, const func_t &f) {
    if (iter.numel() == 0) {
        return;
    }
    if (!iter.can_use_32bit_indexing()) {
        for (auto &sub_iter : iter.with_32bit_indexing()) {
            gpu_kernel(sub_iter, f);
        }
        return;
    }
    gpu_kernel_impl(iter, f);
}

} // namespace pmkl
