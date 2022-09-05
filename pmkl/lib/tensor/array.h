#pragma once

namespace pmkl {
namespace memory {

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T, int vec_size>
struct array {
    T val[vec_size];
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T>
struct alignas(sizeof(T) * 4) vec4 {
    union {
        T val[4];
        struct {
            T x, y, z, w;
        };
    };
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

template <typename T>
struct alignas(sizeof(T) * 2) vec2 {
    union {
        T val[2];
        struct {
            T x, y;
        };
    };
    T &operator[](int i) {
        return val[i];
    }
    T const &operator[](int i) const {
        return val[i];
    }
};

}
} // namespace pmkl::memory