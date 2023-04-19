#pragma once

#include <CL/sycl.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <array>
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

#define GPU_WARP_SIZE 32

namespace pmkl {

using namespace utils;

struct KernelInfo {
    using index_t = const unsigned int; // May change on some device
    sycl::nd_item<3> &item_;
    char *smem_;

    DEVICE_INLINE KernelInfo(sycl::nd_item<3> &item,
                             sycl::local_accessor<char, 1> buffer) :
        item_(item),
        smem_((char *)buffer.get_pointer().get()) {
    }
    DEVICE_INLINE index_t thread_idx(int d) {
        return item_.get_local_id(2 - d);
    }
    DEVICE_INLINE index_t thread_range(int d) {
        return item_.get_local_range(2 - d);
    }
    DEVICE_INLINE index_t block_idx(int d) {
        return item_.get_group(2 - d);
    }
    DEVICE_INLINE index_t block_range(int d) {
        return item_.get_group_range(2 - d);
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
        queues_[current_device_].wait();
    }

    void stream_sync() {
        queues_[current_device_].wait();
    }

    void stream_end() {
        queues_[current_device_].wait();
    }

    void reset_device() {
    }

    void set_device(int d, bool reset = true) {
        CHECK_FAIL(d >= 0 && d < device_count_);
        current_device_ = d;
        if (reset) reset_device();
    }

    int device() const {
        return current_device_;
    }

    template <typename T>
    T *malloc(size_t len) {
        return sycl::aligned_alloc_device<T>(256, len, queues_[current_device_]);
    }

    void free(void *ptr) {
        sycl::free(ptr, queues_[current_device_]);
    }

    void memcpy(void *dst, const void *src, unsigned int len, Direction dir, bool sync = true) {
        queues_[current_device_].memcpy(dst, src, len * sizeof(char));
        if (sync) stream_sync();
    }

    void memset(void *ptr, int value, size_t count, bool sync = true) {
        queues_[current_device_].fill<unsigned char>((unsigned char *)ptr, (unsigned char)value, count);
        if (sync) stream_sync();
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

    int work_size_for_loops() const {
        return max_thread_per_block() / 4;
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

    void set_profiling_mode(bool status) {
        profiling_mode_ = status;
    }

private:
    double get_timems(cl::sycl::event &event) {
        auto submit_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
        auto start_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
        auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
        auto submission_time = (start_time - submit_time) / 1000000.0f;
        auto execution_time = (end_time - start_time) / 1000000.0f;
        return execution_time;
    }

    GpuLauncher() {
        // Need intrinsic API
        srand((unsigned)time(0));
        auto platforms = sycl::platform::get_platforms();
        std::vector<sycl::device> list_devices;
        for (const auto &p : platforms) {
            if (p.get_backend() != sycl::backend::ext_oneapi_level_zero)
                continue;
            auto device_list = p.get_devices();
            for (const auto &device : device_list) {
                if (device.is_gpu()) {
                    queues_.push_back(sycl::queue(device, sycl::property_list{sycl::property::queue::in_order{}, cl::sycl::property::queue::enable_profiling()}));
                    list_devices.push_back(device);
                }
            }
        }
        device_count_ = list_devices.size();
        for (int i = 0; i < device_count_; i++) {
            auto &prop = list_devices[i];
            device_names_.push_back(prop.get_info<sycl::info::device::name>());
            device_max_thread_per_block_.push_back(prop.get_info<sycl::info::device::max_work_group_size>());
            device_shared_memory_.push_back(prop.get_info<sycl::info::device::local_mem_size>());
            device_global_memory_.push_back(prop.get_info<sycl::info::device::global_mem_size>());
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
    std::vector<sycl::queue> queues_;
    bool sync_mode_;
    bool profiling_mode_ = false;

public:
    template <typename func_t, typename... args_t>
    void submit(
        size_t slm_size,
        std::vector<int> grid_size,
        std::vector<int> block_size,
        func_t fn, args_t &&...args) {
        std::array<int, 3> groups, group_items;
        if (block_size.size() == 1) {
            group_items[0] = 1;
            group_items[1] = 1;
            group_items[2] = block_size[0];
        } else if (block_size.size() == 2) {
            group_items[0] = 1;
            group_items[1] = block_size[1];
            group_items[2] = block_size[0];
        } else if (block_size.size() == 3) {
            group_items[0] = block_size[2];
            group_items[1] = block_size[1];
            group_items[2] = block_size[0];
        }
        if (grid_size.size() == 1) {
            groups[0] = group_items[0] * 1;
            groups[1] = group_items[1] * 1;
            groups[2] = group_items[2] * grid_size[0];
        } else if (grid_size.size() == 2) {
            groups[0] = group_items[0] * 1;
            groups[1] = group_items[1] * grid_size[1];
            groups[2] = group_items[2] * grid_size[0];
        } else if (grid_size.size() == 3) {
            groups[0] = group_items[0] * grid_size[2];
            groups[1] = group_items[1] * grid_size[1];
            groups[2] = group_items[2] * grid_size[0];
        }

        auto event = queues_[current_device_].submit([&](sycl::handler &h) {
            auto slm = sycl::local_accessor<char, 1>(slm_size, h);
            h.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(groups[0], groups[1], groups[2]),
                                  sycl::range<3>(group_items[0], group_items[1], group_items[2])),
                [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(GPU_WARP_SIZE)]] {
                    auto info = KernelInfo(item, slm);
                    fn(info, std::forward<args_t>(args)...);
                });
        });
        if (profiling_mode_) {
            std::cout << get_timems(event) << " ms" << std::endl;
        }
        if (is_sync_mode()) stream_sync();
    }
};
GpuLauncher *GpuLauncher::m_pInstance = new GpuLauncher();

template <typename T>
DEVICE_INLINE T GPU_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    std::cout << "NOT IMPLEMENTED ERROR\n";
    return value;
}

template <typename T>
DEVICE_INLINE T GPU_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    std::cout << "NOT IMPLEMENTED ERROR\n";
    return value;
}

template <typename T>
DEVICE_INLINE T GPU_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff) {
    std::cout << "NOT IMPLEMENTED ERROR\n";
    return value;
}

}; // namespace pmkl
