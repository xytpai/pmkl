#pragma once

#include "launcher.h"
#include "tensor.h"
#include "measure.h"

namespace pmkl {

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

#define DivUp(A, B) (((A) + (B)-1) / (B))
#define NUMERIC_MAX(A, B) (((A) > (B)) ? (A) : (B))
#define NUMERIC_MIN(A, B) (((A) > (B)) ? (B) : (A))

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
    enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

} // namespace pmkl
