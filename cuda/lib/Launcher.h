#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>

int perr_(int ret, const char *errinfo) {
    if (ret) {
        printf("PERR(%d): %s\n", ret, errinfo);
        return ret;
    }
    return 0;
}
#define PERR(FUNC, ...) perr_(FUNC(__VA_ARGS__), #FUNC)

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

    int GetCount() const {
        return device_count_;
    }

    int SetDevice(int d) {
        if (d < 0 || d >= device_count_) {
            return perr_(-1, "Device invalid");
        }
        if (stream) PERR(cudaStreamDestroy, (cudaStream_t)stream);
        stream = 0;
        current_device_ = d;
        int ret = PERR(cudaSetDevice, current_device_);
        ret |= PERR(cudaDeviceReset);
        return ret;
    }

    int ResetDevice() {
        return PERR(cudaDeviceReset);
    }

    std::string GetDeviceName() const {
        return device_names_[current_device_];
    }

    int GetMaxThreadPerBlock() const {
        return device_max_thread_per_block_[current_device_];
    }

    long GetSharedMemory() const {
        return device_shared_memory_[current_device_];
    }

    long GetGlobalMemory() const {
        return device_global_memory_[current_device_];
    }

    template <typename DataType>
    DataType *Malloc(unsigned int len) {
        DataType *ptr = nullptr;
        if (PERR(cudaMalloc, (void **)&ptr, len * sizeof(DataType))) {
            cudaFree((void *)ptr);
            return nullptr;
        }
        return ptr;
    }

    template <typename DataType>
    int Free(DataType *ptr) {
        if (ptr == nullptr) return 0;
        int ret = PERR(cudaFree, ptr);
        ptr = nullptr;
        return ret;
    }

    int StreamBegin() {
        return PERR(cudaStreamCreate, (cudaStream_t *)&stream);
    }

    int StreamSync() {
        return PERR(cudaStreamSynchronize, (cudaStream_t)stream);
    }

    int StreamEnd() {
        int ret = PERR(cudaStreamDestroy, (cudaStream_t)stream);
        if (ret) return ret;
        stream = 0;
        return 0;
    }

    size_t GetStream() const {
        return stream;
    }

    template <typename DataType>
    int Memcpy(DataType *dst, DataType *src, unsigned int len, Direction dir) {
#define GPU_D2H cudaMemcpyDeviceToHost
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2D cudaMemcpyDeviceToDevice
        int ret = 0;
        switch (dir) {
        case Direction::H2D:
            ret |= StreamBegin();
            ret |= PERR(cudaMemcpyAsync, dst, src, len * sizeof(DataType), GPU_H2D, (cudaStream_t)stream);
            ret |= StreamSync();
            ret |= StreamEnd();
            break;
        case Direction::D2H:
            ret |= StreamBegin();
            ret |= PERR(cudaMemcpyAsync, dst, src, len * sizeof(DataType), GPU_D2H, (cudaStream_t)stream);
            ret |= StreamSync();
            ret |= StreamEnd();
            break;
        case Direction::D2D:
            ret |= StreamBegin();
            ret |= PERR(cudaMemcpyAsync, dst, src, len * sizeof(DataType), GPU_D2D, (cudaStream_t)stream);
            ret |= StreamSync();
            ret |= StreamEnd();
            break;
        default:
            ret = perr_(-1, "Direction invalid");
        }
        return ret;
    }

private:
    GpuLauncher() {
        srand((unsigned)time(0));
        cudaGetDeviceCount(&device_count_);
        for (int i = 0; i < device_count_; i++) {
            cudaSetDevice(i);
            int dev;
            cudaDeviceProp prop;
            if (PERR(cudaGetDevice, &dev)) return;
            if (PERR(cudaGetDeviceProperties, &prop, dev)) return;
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
        SetDevice(0);
    }

    ~GpuLauncher() {
        SetDevice(0); // reset state
    }

    GpuLauncher(const GpuLauncher &) = delete;
    GpuLauncher &operator=(const GpuLauncher &) = delete;

    static GpuLauncher *m_pInstance;

    int device_count_;
    std::vector<std::string> device_names_;
    std::vector<int> device_max_thread_per_block_;
    std::vector<long> device_shared_memory_;
    std::vector<long> device_global_memory_;
    int current_device_;
    size_t stream;
};
GpuLauncher *GpuLauncher::m_pInstance = new GpuLauncher();

#define DISPATCH_KERNEL(STREAM, KERNEL, GRID_SIZES, BLOCK_SIZES, SLM_SIZE, ...) \
    {                                                                           \
        dim3 grid, block;                                                       \
        if (GRID_SIZES.size() == 1)                                             \
            grid = dim3(GRID_SIZES[0]);                                         \
        else if (GRID_SIZES.size() == 2)                                        \
            grid = dim3(GRID_SIZES[0], GRID_SIZES[1]);                          \
        else if (GRID_SIZES.size() == 3)                                        \
            grid = dim3(GRID_SIZES[0], GRID_SIZES[1], GRID_SIZES[2]);           \
        if (BLOCK_SIZES.size() == 1)                                            \
            block = dim3(BLOCK_SIZES[0]);                                       \
        else if (BLOCK_SIZES.size() == 2)                                       \
            block = dim3(BLOCK_SIZES[0], BLOCK_SIZES[1]);                       \
        else if (BLOCK_SIZES.size() == 3)                                       \
            block = dim3(BLOCK_SIZES[0], BLOCK_SIZES[1], BLOCK_SIZES[2]);       \
        KERNEL<<<grid, block, SLM_SIZE, (cudaStream_t)STREAM>>>(__VA_ARGS__);   \
    }

#define DISPATCH_KERNEL_TEST(KERNEL, GRID_SIZES, BLOCK_SIZES, SLM_SIZE, ...) \
    {                                                                        \
        cudaEvent_t start, stop;                                             \
        cudaEventCreate(&start);                                             \
        cudaEventCreate(&stop);                                              \
        auto launcher = GpuLauncher::GetInstance();                          \
        launcher->StreamBegin();                                             \
        cudaEventRecord(start);                                              \
        DISPATCH_KERNEL(                                                     \
            launcher->GetStream(),                                           \
            KERNEL,                                                          \
            GRID_SIZES,                                                      \
            BLOCK_SIZES,                                                     \
            SLM_SIZE,                                                        \
            __VA_ARGS__);                                                    \
        cudaEventRecord(stop);                                               \
        launcher->StreamSync();                                              \
        cudaEventSynchronize(stop);                                          \
        launcher->StreamEnd();                                               \
        float milliseconds = 0;                                              \
        cudaEventElapsedTime(&milliseconds, start, stop);                    \
        std::cout << #KERNEL << ": " << milliseconds << " ms" << std::endl;  \
    }
