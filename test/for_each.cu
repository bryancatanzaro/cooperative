#include <cooperative/cuda/tag.h>
#include <cooperative/cuda/functional.h>
#include <cooperative/cuda/transform.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

#include <iostream>

//Requires power of 2 thread block size
//Consider how to encode this in the type system
//(We want several kinds of constraints: exact size, power of two
//size, bounded size...)
template<typename T>
struct multiply_by_reduction
    : public cooperative::cuda::unary_cooperative_function<T, T> {
    typedef cooperative::cuda::unary_cooperative_function<T, T> super_t;
    __host__
    multiply_by_reduction()
        : super_t(cooperative::cuda::group_config(64, 64 * sizeof(T))) {}
    
    using cooperative::cuda::unary_cooperative_function<T, T>::g;
    __device__
    T operator()(const T& v) const {
        T* scratch = g.template scratch<T>();
        int local_id = g.local_thread_id();
        int group_size = g.group_size();
        scratch[local_id] = v;
        g.barrier();
        int offset = group_size >> 1;
        while(offset > 0) {
            if (local_id < offset)
                scratch[local_id] += scratch[local_id+offset];
            g.barrier();
            offset >>= 1;
        }
        return scratch[local_id];
    }

};


int main() {
    thrust::counting_iterator<int> x;
    int length = 100;
    thrust::device_vector<int> y(length);
    thrust::transform(thrust::retag<cooperative::cuda::tag>(x),
                      thrust::retag<cooperative::cuda::tag>(x+length),
                      thrust::retag<cooperative::cuda::tag>(y.begin()),
                      multiply_by_reduction<int>());


    thrust::copy(y.begin(), y.end(), std::ostream_iterator<int>(std::cout, "\n"));
                     
    
}
