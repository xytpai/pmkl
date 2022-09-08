#pragma once

#include "common.h"

namespace pmkl {

template <typename Type>
struct KeyTraits {};

template <>
struct KeyTraits<NullType> {
    using Type = uint32_t;
    GPU_CODE static inline Type convert(float v) {
        return 0;
    }
    GPU_CODE static inline NullType deconvert(Type v) {
        return NullType();
    }
};

template <>
struct KeyTraits<float> {
    using Type = uint32_t;
    GPU_CODE static inline Type convert(float v) {
        Type x = *((Type *)&v);
        Type mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (x ^ mask);
    }
    GPU_CODE static inline float deconvert(Type v) {
        Type mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
        auto v_de = v ^ mask;
        return *((float *)&v_de);
    }
};

template <>
struct KeyTraits<uint8_t> {
    using Type = uint8_t;
    GPU_CODE static inline Type convert(uint8_t v) {
        return v;
    }
    GPU_CODE static inline uint8_t deconvert(Type v) {
        return v;
    }
};

template <>
struct KeyTraits<int8_t> {
    using Type = int8_t;
    GPU_CODE static inline Type convert(int8_t v) {
        return 128u + v;
    }
    GPU_CODE static inline int8_t deconvert(Type v) {
        return v - 128;
    }
};

template <>
struct KeyTraits<int16_t> {
    using Type = int16_t;
    GPU_CODE static inline Type convert(int16_t v) {
        return 32768u + v;
    }
    GPU_CODE static inline int16_t deconvert(Type v) {
        return v - 32768;
    }
};

template <>
struct KeyTraits<int32_t> {
    using Type = uint32_t;
    GPU_CODE static inline Type convert(int32_t v) {
        return 2147483648u + v;
    }
    GPU_CODE static inline int32_t deconvert(Type v) {
        return v - 2147483648u;
    }
};

template <>
struct KeyTraits<int64_t> {
    using Type = uint64_t;
    GPU_CODE static inline Type convert(int64_t v) {
        return 9223372036854775808ull + v;
    }
    GPU_CODE static inline int64_t deconvert(Type v) {
        return v - 9223372036854775808ull;
    }
};

template <>
struct KeyTraits<double> {
    using Type = uint64_t;
    GPU_CODE static inline Type convert(double v) {
        Type x = *((Type *)&v);
        Type mask = -((x >> 63)) | 0x8000000000000000;
        return (x ^ mask);
    }
    GPU_CODE static inline double deconvert(Type v) {
        Type mask = ((v >> 63) - 1) | 0x8000000000000000;
        auto v_de = v ^ mask;
        return *((double *)&v_de);
    }
};

} // namespace pmkl