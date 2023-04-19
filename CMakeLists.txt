cmake_minimum_required(VERSION 3.0)
project(pmkl)
set(ROOTDIR ${PROJECT_SOURCE_DIR})

include_directories(${ROOTDIR}/pmkl)

include_directories(${ROOTDIR}/pmkl/core)
include_directories(${ROOTDIR}/pmkl/core/launcher)
include_directories(${ROOTDIR}/pmkl/core/memory)
include_directories(${ROOTDIR}/pmkl/core/utils)

include_directories(${ROOTDIR}/pmkl/eltwise)
include_directories(${ROOTDIR}/pmkl/norm)
include_directories(${ROOTDIR}/pmkl/sorting)
include_directories(${ROOTDIR}/pmkl/image/nms)


add_definitions(-DDEBUG)
enable_testing()


find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    message("Cuda include path: ${CUDA_INCLUDE_DIRS}")

    aux_source_directory(${ROOTDIR}/test_cu DEV_SRCS)
    message("Source files: ${DEV_SRCS}")

    list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "--std=c++17")
    foreach(EXAMPLE ${DEV_SRCS})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_EXAMPLE ${EXAMPLE})
        cuda_add_executable(${FILE_EXAMPLE} ${EXAMPLE} ${DIR_SRCS})
        add_test(NAME ${FILE_EXAMPLE} COMMAND ${FILE_EXAMPLE})
        message("example ${FILE_EXAMPLE}")
    endforeach()
endif(CUDA_FOUND)


find_package(IntelDPCPP)
if(IntelDPCPP_FOUND)
    add_definitions(-DUSE_DPCPP)

    aux_source_directory(${ROOTDIR}/test DEV_SRCS)
    message("Source files: ${DEV_SRCS}")

    foreach(EXAMPLE ${DEV_SRCS})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_EXAMPLE ${EXAMPLE})
        add_executable(${FILE_EXAMPLE} ${EXAMPLE} ${DIR_SRCS})
        add_test(NAME ${FILE_EXAMPLE} COMMAND ${FILE_EXAMPLE})
        message("example ${FILE_EXAMPLE}")
    endforeach()

endif(IntelDPCPP_FOUND)
