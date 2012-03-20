#pragma once
#include <thrust/system/cuda/memory.h>

namespace cooperative {
namespace cuda {

struct tag : thrust::system::cuda::tag{};

tag select_system(tag,tag)
{
  return tag();
}

}
}
