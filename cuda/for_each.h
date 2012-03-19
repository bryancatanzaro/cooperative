#pragma once
#include <thrust/system/cuda/memory.h>


namespace cooperative {
namespace cuda {

template<typename Iterator, typename Function>
Iterator for_each(tag, Iterator first, Iterator last, Function f)
{
  std::cout << "for_each cooperative::cuda::tag" << std::endl;

  thrust::for_each(thrust::retag<thrust::system::cuda::tag>(first),
                   thrust::retag<thrust::system::cuda::tag>(last),
                   f);

  return last;
}


}
}
