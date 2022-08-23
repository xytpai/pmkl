#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "exception.h"

namespace pmkl {

class GpuLauncher {
public:
    enum Direction {
        D2H = 0,
        H2D,
        D2D
    };

    static GpuLauncher *
    GetInstance() {
        return m_pInstance;
    }

    // Intrinsic API for device control

    void stream_begin() {
        CHECK_FAIL(cudaStreamCreate((cudaStream_t *)&stream) == 0);
    }

    void stream_sync() {
        CHECK_FAIL(cudaStreamSynchronize((cudaStream_t)stream) == 0);
    }

    void stream_end() {
        CHECK_FAIL(cudaStreamDestroy((cudaStream_t)stream) == 0);
        stream = 0;
    }

    void reset_device() {
        CHECK_FAIL(cudaDeviceReset() == 0);
    }

    void set_device(int d, bool reset = true) {
        CHECK_FAIL(d >= 0 && d < device_count_);
        if (stream) stream_end();
        current_device_ = d;
        CHECK_FAIL(cudaSetDevice(current_device_) == 0);
        if (reset) reset_device();
    }

    template <typename T>
    T *malloc(size_t len) {
        T *ptr = nullptr;
        CHECK_FAIL(cudaMalloc((void **)&ptr, len * sizeof(T)) == 0);
        return ptr;
    }

    template <typename T>
    void free(T *ptr) {
        if (ptr) CHECK_FAIL(cudaFree(ptr) == 0);
    }

    template <typename T>
    void memcpy(T *dst, const T *src, unsigned int len, Direction dir, bool sync = true) {
        switch (dir) {
        case Direction::H2D:
            stream_begin();
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream)
                       == 0);
            stream_sync();
            stream_end();
            break;
        case Direction::D2H:
            stream_begin();
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream)
                       == 0);
            stream_sync();
            stream_end();
            break;
        case Direction::D2D:
            stream_begin();
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyDeviceToDevice, (cudaStream_t)stream)
                       == 0);
            stream_sync();
            stream_end();
            break;
        default:
            CHECK_FAIL(false, "invalid direction");
        }
    }

    // For property

    int get_count() const {
        return device_count_;
    }

    size_t get_stream() const {
        return stream;
    }

    std::string get_device_name() const {
        return device_names_[current_device_];
    }

    int get_mak_thread_per_block() const {
        return device_max_thread_per_block_[current_device_];
    }

    size_t get_shared_local_memory_size() const {
        return device_shared_memory_[current_device_];
    }

    size_t get_global_memory_size() const {
        return device_global_memory_[current_device_];
    }

private:
    GpuLauncher() {
        // Need intrinsic API
        srand((unsigned)time(0));
        CHECK_FAIL(cudaGetDeviceCount(&device_count_) == 0);
        for (int i = 0; i < device_count_; i++) {
            set_device(i, false);
            int dev;
            cudaDeviceProp prop;
            CHECK_FAIL(cudaGetDevice(&dev) == 0);
            CHECK_FAIL(cudaGetDeviceProperties(&prop, dev) == 0);
            device_names_.push_back(prop.name);
            device_max_thread_per_block_.push_back(prop.maxThreadsPerBlock);
            device_shared_memory_.push_back(prop.sharedMemPerBlock);
            device_global_memory_.push_back(prop.totalGlobalMem);
        }
#ifdef DEBUG
        std::cout << "Device count: " << device_count_ << std::endl
                  << std::endl;
        for (int i = 0; i < device_count_; i++) {
            std::cout << "[" << i << "]Name: " << device_names_[i] << std::endl;
            std::cout << "MaxThreadPerBlock: " << device_max_thread_per_block_[i] << std::endl;
            std::cout << "SharedMemory(KB): " << device_shared_memory_[i] / 1024 << std::endl;
            std::cout << "GlobalMemory(MB): " << device_global_memory_[i] / 1024 / 1024 << std::endl;
        }
        std::cout << std::endl;
#endif
        set_device(0);
    }

    ~GpuLauncher() {
        set_device(0); // reset state
    }

    GpuLauncher(const GpuLauncher &) = delete;
    GpuLauncher &operator=(const GpuLauncher &) = delete;

    static GpuLauncher *m_pInstance;

    int device_count_;
    std::vector<std::string> device_names_;
    std::vector<int> device_max_thread_per_block_;
    std::vector<size_t> device_shared_memory_;
    std::vector<size_t> device_global_memory_;
    int current_device_;
    size_t stream;
};
GpuLauncher *GpuLauncher::m_pInstance = new GpuLauncher();

// #define DISPATCH_KERNEL(STREAM, KERNEL, GRID_SIZES, BLOCK_SIZES, SLM_SIZE, ...) \
//     {                                                                           \
//         dim3 grid, block;                                                       \
//         if (GRID_SIZES.size() == 1)                                             \
//             grid = dim3(GRID_SIZES[0]);                                         \
//         else if (GRID_SIZES.size() == 2)                                        \
//             grid = dim3(GRID_SIZES[0], GRID_SIZES[1]);                          \
//         else if (GRID_SIZES.size() == 3)                                        \
//             grid = dim3(GRID_SIZES[0], GRID_SIZES[1], GRID_SIZES[2]);           \
//         if (BLOCK_SIZES.size() == 1)                                            \
//             block = dim3(BLOCK_SIZES[0]);                                       \
//         else if (BLOCK_SIZES.size() == 2)                                       \
//             block = dim3(BLOCK_SIZES[0], BLOCK_SIZES[1]);                       \
//         else if (BLOCK_SIZES.size() == 3)                                       \
//             block = dim3(BLOCK_SIZES[0], BLOCK_SIZES[1], BLOCK_SIZES[2]);       \
//         KERNEL<<<grid, block, SLM_SIZE, (cudaStream_t)STREAM>>>(__VA_ARGS__);   \
//     }

// #define DISPATCH_KERNEL_TEST(KERNEL, GRID_SIZES, BLOCK_SIZES, SLM_SIZE, ...) \
//     {                                                                        \
//         cudaEvent_t start, stop;                                             \
//         cudaEventCreate(&start);                                             \
//         cudaEventCreate(&stop);                                              \
//         auto launcher = GpuLauncher::GetInstance();                          \
//         launcher->StreamBegin();                                             \
//         cudaEventRecord(start);                                              \
//         DISPATCH_KERNEL(                                                     \
//             launcher->GetStream(),                                           \
//             KERNEL,                                                          \
//             GRID_SIZES,                                                      \
//             BLOCK_SIZES,                                                     \
//             SLM_SIZE,                                                        \
//             __VA_ARGS__);                                                    \
//         cudaEventRecord(stop);                                               \
//         launcher->StreamSync();                                              \
//         cudaEventSynchronize(stop);                                          \
//         launcher->StreamEnd();                                               \
//         float milliseconds = 0;                                              \
//         cudaEventElapsedTime(&milliseconds, start, stop);                    \
//         std::cout << #KERNEL << ": " << milliseconds << " ms" << std::endl;  \
//     }
}; // namespace pmkl