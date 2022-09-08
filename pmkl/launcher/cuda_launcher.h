#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "exception.h"

#ifndef HOST_DEVICE
#define HOST_DEVICE __host__ __device__
#endif

#ifndef HOST
#define HOST __host__
#endif

#ifndef DEVICE
#define DEVICE __device__
#endif

#ifndef HOST_DEVICE_INLINE
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#endif

#ifndef DEVICE_LAMBDA
#define DEVICE_LAMBDA __host__ __device__
#endif

namespace pmkl {

using namespace utils;

struct KernelInfo {
    using index_t = const unsigned int; // May change on some device
    index_t &t_size0, &t_size1, &t_size2;
    index_t &t0, &t1, &t2;
    index_t &b_size0, &b_size1, &b_size2;
    index_t &b0, &b1, &b2;
    HOST_DEVICE_INLINE KernelInfo(
        index_t &t_size0, index_t &t_size1, index_t &t_size2,
        index_t &t0, index_t &t1, index_t &t2,
        index_t &b_size0, index_t &b_size1, index_t &b_size2,
        index_t &b0, index_t &b1, index_t &b2) :
        t_size0(t_size0),
        t_size1(t_size1), t_size2(t_size2),
        t0(t0), t1(t1), t2(t2),
        b_size0(b_size0), b_size1(b_size1), b_size2(b_size2),
        b0(b0), b1(b1), b2(b2) {
    }
};

template <typename func_t, typename... args_t>
__global__ void kernel_wrapper(func_t fn, args_t... args) {
    auto info = KernelInfo(
        blockDim.x, blockDim.y, blockDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z,
        gridDim.x, gridDim.y, gridDim.z,
        blockIdx.x, blockIdx.y, blockIdx.z);
    fn(info, std::forward<args_t>(args)...);
}

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
        if (stream_) stream_end();
        CHECK_FAIL(cudaStreamCreate((cudaStream_t *)&stream_) == 0);
    }

    void stream_sync() {
        if (stream_) CHECK_FAIL(cudaStreamSynchronize((cudaStream_t)stream_) == 0);
    }

    void stream_end() {
        stream_sync();
        CHECK_FAIL(cudaStreamDestroy((cudaStream_t)stream_) == 0);
        stream_ = 0;
    }

    void reset_device() {
        CHECK_FAIL(cudaDeviceReset() == 0);
    }

    void set_device(int d, bool reset = true) {
        CHECK_FAIL(d >= 0 && d < device_count_);
        if (stream_) stream_end();
        current_device_ = d;
        CHECK_FAIL(cudaSetDevice(current_device_) == 0);
        if (reset) reset_device();
    }

    int device() const {
        return current_device_;
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
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        switch (dir) {
        case Direction::H2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2H:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(T),
                                       cudaMemcpyDeviceToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        default:
            CHECK_FAIL(false, "invalid direction");
        }
        if (sync) stream_sync();
        if (need_new_stream) stream_end();
    }

    void memset(void *ptr, int value, size_t count, bool sync = true) {
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        cudaMemsetAsync(ptr, value, count, (cudaStream_t)stream_);
        if (sync) stream_sync();
        if (need_new_stream) stream_end();
    }

    // For property

    int device_count() const {
        return device_count_;
    }
    size_t stream() const {
        return stream_;
    }

    std::string device_names() const {
        return device_names_[current_device_];
    }

    int max_thread_per_block() const {
        return device_max_thread_per_block_[current_device_];
    }

    size_t shared_local_memory_size() const {
        return device_shared_memory_[current_device_];
    }

    size_t global_memory_size() const {
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
    size_t stream_;

public:
    template <typename func_t, typename... args_t>
    void submit(
        size_t slm_size,
        std::vector<int> grid_size,
        std::vector<int> block_size,
        func_t fn, args_t &&... args) {
        dim3 grid, block;
        if (grid_size.size() == 1)
            grid = dim3(grid_size[0]);
        else if (grid_size.size() == 2)
            grid = dim3(grid_size[0], grid_size[1]);
        else if (grid_size.size() == 3)
            grid = dim3(grid_size[0], grid_size[1], grid_size[2]);
        if (block_size.size() == 1)
            block = dim3(block_size[0]);
        else if (block_size.size() == 2)
            block = dim3(block_size[0], block_size[1]);
        else if (block_size.size() == 3)
            block = dim3(block_size[0], block_size[1], block_size[2]);

        kernel_wrapper<<<grid, block, slm_size, (cudaStream_t)stream_>>>(fn,
                                                                         std::forward<args_t>(args)...);
    }
};
GpuLauncher *GpuLauncher::m_pInstance = new GpuLauncher();

}; // namespace pmkl