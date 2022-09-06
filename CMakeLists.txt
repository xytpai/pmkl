cmake_minimum_required(VERSION 3.0)
project(pmkl)
set(ROOTDIR ${PROJECT_SOURCE_DIR})

aux_source_directory(${ROOTDIR}/pmkl DEV_SRCS)
message("Source files: ${DEV_SRCS}")

include_directories(${ROOTDIR}/pmkl)
include_directories(${ROOTDIR}/pmkl/utils)
include_directories(${ROOTDIR}/pmkl/tensor)

add_definitions(-DDEBUG)

find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
    message("Cuda include path: ${CUDA_INCLUDE_DIRS}")
    foreach(EXAMPLE ${DEV_SRCS})
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_EXAMPLE ${EXAMPLE})
        cuda_add_executable(${FILE_EXAMPLE} ${EXAMPLE} ${DIR_SRCS})
    endforeach()
endif(CUDA_FOUND)
