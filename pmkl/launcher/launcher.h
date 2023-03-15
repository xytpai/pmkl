#pragma once

#if defined(USE_CUDA)
#include "cuda_launcher.h"
#endif

#if defined(USE_DPCPP)
#include "sycl_launcher.h"
#endif
