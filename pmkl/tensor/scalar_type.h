#pragma once

#include <iostream>
#include <type_traits>

#include "launcher.h"

namespace pmkl {

#define FORALL_BASIC_SCALAR_TYPES(_) \
    _(uint8_t, Byte)  /* 0 */        \
    _(int8_t, Char)   /* 1 */        \
    _(int16_t, Short) /* 2 */        \
    _(int, Int)       /* 3 */        \
    _(int64_t, Long)  /* 4 */        \
    _(float, Float)   /* 5 */        \
    _(double, Double) /* 6 */        \
    _(bool, Bool)     /* 7 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
    FORALL_BASIC_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
        Undefined,
    NumOptions
};

static inline const char *to_string(ScalarType t) {
#define DEFINE_CASE(_, name) \
    case ScalarType::name:   \
        return #name;

    switch (t) {
        FORALL_BASIC_SCALAR_TYPES(DEFINE_CASE)
    default:
        return "UNKNOWN_SCALAR";
    }
#undef DEFINE_CASE
}

static inline size_t element_size(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
    case ScalarType::name:                 \
        return sizeof(ctype);

    switch (t) {
        FORALL_BASIC_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
        CHECK_FAIL(false, "Unknown ScalarType");
    }
#undef CASE_ELEMENTSIZE_CASE
    return 0;
}

std::ostream &operator<<(std::ostream &os, const ScalarType &dtype) {
    os << to_string(dtype);
    return os;
}

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)          \
    template <>                                                        \
    struct CppTypeToScalarType<cpp_type>                               \
        : std::                                                        \
              integral_constant<ScalarType, ScalarType::scalar_type> { \
    };

FORALL_BASIC_SCALAR_TYPES(SPECIALIZE_CppTypeToScalarType)

#define FETCH_AND_CAST_CASE(type, scalartype) \
    case ScalarType::scalartype:              \
        return static_cast<type>(*(const type *)ptr);
template <typename dest_t>
HOST_DEVICE_INLINE dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
        FORALL_BASIC_SCALAR_TYPES(FETCH_AND_CAST_CASE)
    default:;
    }
    return dest_t(0);
}
#undef FETCH_AND_CAST_CASE

#define CAST_AND_STORE_CASE(type, scalartype)    \
    case ScalarType::scalartype:                 \
        *(type *)ptr = static_cast<type>(value); \
        return;
template <typename src_t>
HOST_DEVICE_INLINE void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
    switch (dest_type) {
        FORALL_BASIC_SCALAR_TYPES(CAST_AND_STORE_CASE)
    default:;
    }
}
#undef CAST_AND_STORE_CASE

#undef FORALL_BASIC_SCALAR_TYPES
#undef SPECIALIZE_CppTypeToScalarType

} // namespace pmkl