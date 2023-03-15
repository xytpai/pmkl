#pragma once

#include <CL/sycl.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "exception.h"

#ifndef HOST_DEVICE
#define HOST_DEVICE 
#endif

#ifndef HOST
#define HOST 
#endif

#ifndef DEVICE
#define DEVICE 
#endif

#ifndef HOST_DEVICE_INLINE
#define HOST_DEVICE_INLINE inline
#endif

#ifndef DEVICE_INLINE
#define DEVICE_INLINE inline
#endif

namespace pmkl {

using namespace utils;

struct KernelInfo {
    using index_t = const unsigned int; // May change on some device
    sycl::nd_item<3> &item_;
    char *smem_;
    
    DEVICE_INLINE KernelInfo(sycl::nd_item<3> &item, 
        sycl::local_accessor<unsigned char, 1> &buffer) :
        item_(item) {
        smem_ = reinterpret_cast<char*>(buffer.get_pointer().get());
    }
    DEVICE_INLINE index_t thread_idx(int d) {
        switch (d) {
        case 0:
            return item_.get_local_id(2);
        case 1:
            return item_.get_local_id(1);
        case 2:
            return item_.get_local_id(0);
        }
        return 0;
    }
    DEVICE_INLINE index_t thread_range(int d) {
        switch (d) {
        case 0:
            return item_.get_local_range(2);
        case 1:
            return item_.get_local_range(1);
        case 2:
            return item_.get_local_range(0);
        }
        return 0;
    }
    DEVICE_INLINE index_t block_idx(int d) {
        switch (d) {
        case 0:
            return item_.get_group(2);
        case 1:
            return item_.get_group(1);
        case 2:
            return item_.get_group(0);
        }
        return 0;
    }
    DEVICE_INLINE index_t block_range(int d) {
        switch (d) {
        case 0:
            return item_.get_group_range(2);
        case 1:
            return item_.get_group_range(1);
        case 2:
            return item_.get_group_range(0);
        }
        return 0;
    }
    DEVICE_INLINE index_t group_idx(int d) {
        return block_idx(d);
    }
    DEVICE_INLINE index_t group_range(int d) {
        return block_range(d);
    }
    DEVICE_INLINE void barrier() {
        item_.barrier(sycl::access::fence_space::local_space);
    }
    DEVICE_INLINE char *shared_ptr() {
        return smem_;
    };
};

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
        // if (stream_) stream_end();
        // CHECK_FAIL(cudaStreamCreate((cudaStream_t *)&stream_) == 0);
        stream_ = sycl::queue(sycl::gpu_selector{}, sycl::property_list{sycl::property::queue::in_order{}});
    }

    void stream_sync() {
        stream_.wait();
        // if (stream_) CHECK_FAIL(cudaStreamSynchronize((cudaStream_t)stream_) == 0);
    }

    void stream_end() {
        stream_.wait();
        // stream_sync();
        // CHECK_FAIL(cudaStreamDestroy((cudaStream_t)stream_) == 0);
        // stream_ = 0;
    }

    void reset_device() {
        // CHECK_FAIL(cudaDeviceReset() == 0);
    }

    void set_device(int d, bool reset = true) {
        CHECK_FAIL(d >= 0 && d < device_count_);
        if (stream_) stream_end();
        current_device_ = d;
        // CHECK_FAIL(cudaSetDevice(current_device_) == 0);
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

    void free(void *ptr) {
        if (ptr) CHECK_FAIL(cudaFree(ptr) == 0);
    }

    void memcpy(void *dst, const void *src, unsigned int len, Direction dir, bool sync = true) {
        bool need_new_stream = stream_ != 0 ? false : true;
        if (need_new_stream) stream_begin();
        switch (dir) {
        case Direction::H2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
                                       cudaMemcpyHostToDevice, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2H:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
                                       cudaMemcpyDeviceToHost, (cudaStream_t)stream_)
                       == 0);
            break;
        case Direction::D2D:
            CHECK_FAIL(cudaMemcpyAsync(dst, src, len * sizeof(char),
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

    std::string device_names() const {
        return device_names_[current_device_];
    }

    int max_thread_per_block() const {
        return device_max_thread_per_block_[current_device_];
    }

    int max_thread_per_group() const {
        return max_thread_per_block();
    }

    size_t shared_local_memory_size() const {
        return device_shared_memory_[current_device_];
    }

    size_t global_memory_size() const {
        return device_global_memory_[current_device_];
    }

    void set_sync_mode(bool is_sync) {
        sync_mode_ = is_sync;
    }

    bool is_sync_mode() const {
        return sync_mode_;
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
        sync_mode_ = true;
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
    sycl::queue stream_;
    bool sync_mode_;

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