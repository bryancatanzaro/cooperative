#pragma once
#include <thrust/system/cuda/memory.h>

namespace cooperative {
namespace cuda {

struct tag : thrust::system::cuda::tag{};

}
}
