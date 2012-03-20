#pragma once
#include <thrust/system/cuda/memory.h>
#include <thrust/detail/static_assert.h>
#include <thrust/distance.h>
#include <cooperative/cuda/thread_group.h>
#include <cooperative/group_config.h>
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#include <thrust/detail/minmax.h>

namespace cooperative {
namespace cuda {

template<typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction,
         typename Context>
struct for_each_n_closure
{
    typedef void result_type;
    typedef Context context_type;
    
    RandomAccessIterator first;
    Size n;
    UnaryFunction f;
    Context context;

    for_each_n_closure(RandomAccessIterator _first,
                       Size _n,
                       UnaryFunction _f,
                       Context _context = Context())
        : first(_first), n(_n), f(_f), context(_context) {}

    __device__ __thrust_forceinline__
    result_type operator()(void) {
        extern __shared__ char scratch[];
        thread_group g(scratch, f.get_config().scratch_size);
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


template<typename Iterator, typename Function>
Iterator for_each(tag, Iterator first, Iterator last, Function f)
{
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  THRUST_STATIC_ASSERT( (thrust::detail::depend_on_instantiation<Iterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  size_t n = thrust::distance(first, last);
  
  if (n <= 0) return first;  //empty range

  typedef thrust::system::cuda::detail::detail::blocked_thread_array Context;
  typedef for_each_n_closure<Iterator, int, Function, Context> Closure;

  Closure closure(first, n, f);

  
  //this calculator is probably not optimal for us
  //But at least it should be valid
  thrust::system::cuda::detail::detail::launch_calculator<Closure> calculator;
  thrust::tuple<size_t, size_t, size_t> config = calculator.with_variable_block_size();
  size_t max_blocks = thrust::get<0>(config);
  size_t n_blocks = thrust::min(max_blocks, n);
  size_t n_threads = f.get_config().group_size;
  size_t n_scratch = f.get_config().scratch_size;
  
  thrust::system::cuda::detail::detail::launch_closure_by_value<Closure>
      <<<n_blocks, n_threads, n_scratch>>>(
          closure);
  
  
  
  return last;
}


}
}
