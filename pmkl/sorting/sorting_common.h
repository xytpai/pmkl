#pragma once

#include "launcher.h"

namespace pmkl {
namespace sorting {

template <typename T>
DEVICE_INLINE T NUMERIC_MIN(T A, T B) {
    return (((A) > (B)) ? (B) : (A));
}

template <typename T>
DEVICE_INLINE T DivUp(T A, T B) {
    return (((A) + (B)-1) / (B));
}

struct NullType {
    using value_type = NullType;
    template <typename T>
    HOST_DEVICE_INLINE NullType &operator=(const T &) {
        return *this;
    }
    HOST_DEVICE_INLINE bool operator==(const NullType &) {
        return true;
    }
    HOST_DEVICE_INLINE bool operator!=(const NullType &) {
        return false;
    }
};

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

}
} // namespace pmkl::sorting

#include "key_traits.h"
#include "block_aligned_cumsum.h"
