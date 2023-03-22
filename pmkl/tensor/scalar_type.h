#pragma once

#include <iostream>

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

#undef FORALL_BASIC_SCALAR_TYPES

std::ostream &operator<<(std::ostream &os, const ScalarType &dtype) {
    os << to_string(dtype);
    return os;
}

} // namespace pmkl