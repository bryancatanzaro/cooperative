#pragma once
#include <thrust/system/cuda/memory.h>
#include <thrust/detail/static_assert.h>
#include <thrust/distance.h>
#include <cooperative/cuda/thread_group.h>
namespace cooperative {
namespace cuda {

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
struct for_each_n_closure
{
    typedef void result_type;

    RandomAccessIterator first;
    Size n;
    UnaryFunction f;

    for_each_n_closure(RandomAccessIterator _first,
                       Size _n,
                       UnaryFunction _f)
        : first(_first), n(_n), f(_f) {}

    __device__ __thrust_forceinline__
    result_type operator()(void) {
        extern __shared__ char scratch[];
        //XXX Forward scratch size
        thread_group g(scratch, /*dummy*/0);
        f.set_group(g);
        Size i = g.group_id();
        int number_of_groups = g.number_of_groups();
        
        first += i;
        while(i < n) {
            f(*first);
            i += number_of_groups;
            first += number_of_groups;
        }
    }
};

template<typename Closure>
__global__ void launch_closure_by_value(Closure f) {
    f();
}

template<typename Iterator, typename Function>
Iterator for_each(tag, Iterator first, Iterator last, Function f)
{
  std::cout << "for_each cooperative::cuda::tag" << std::endl;
   // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Iterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  size_t n = thrust::distance(first, last);
  
  if (n <= 0) return first;  //empty range

  //dummy values
  size_t n_blocks = 30;
  size_t n_threads = 256;
  size_t n_scratch = n_threads * sizeof(int);

  launch_closure_by_value<for_each_n_closure<Iterator,int,Function> >
      <<<n_blocks, n_threads, n_scratch>>>(
          for_each_n_closure<Iterator, int, Function>(first, n, f));
  
  
  
  return last;
}


}
}
