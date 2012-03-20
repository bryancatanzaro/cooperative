#include <cooperative/cuda/tag.h>
#include <cooperative/cuda/functional.h>
#include <cooperative/cuda/transform.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>
#include <cooperative/group_config.h>
#include <cooperative/sequences/uniform_sequence.h>
#include <cooperative/sequences/sequence_iterator.h>
#include <iostream>
#include <thrust/iterator/constant_iterator.h>

template<typename Seq>
struct group_sum
    : public cooperative::cuda::unary_cooperative_function<Seq, typename Seq::value_type> {
    typedef typename Seq::value_type T;
    typedef cooperative::cuda::unary_cooperative_function<Seq, T> super_t;
    __host__
    group_sum()
        : super_t(cooperative::group_config(64, 64 * sizeof(T))) {}
    using super_t::group;
    __device__
    T operator()(const Seq& v) const {
        int local_id = group.local_thread_id();
        int group_size = group.group_size();
        T accumulator(0);
        //Blockwise sequential sum
        for(int i = local_id; i < v.size(); i += group_size) {
            accumulator += v[i];
        }
        //Block parallel sum
        T* scratch = group.template scratch<T>();
        scratch[local_id] = accumulator;
        group.barrier();
        int offset = group_size >> 1;
        while(offset > 0) {
            if (local_id < offset)
                scratch[local_id] += scratch[local_id+offset];
            group.barrier();
            offset >>= 1;
        }
        return scratch[local_id];
    }

};


int main() {
    thrust::constant_iterator<float> x(0.5f/128.0f);
    int outer_length = 128;
    int inner_length = 128;
    int overall_length = outer_length * inner_length;
    thrust::device_vector<float> y(overall_length);
    thrust::copy(x, x + overall_length, y.begin());

    cooperative::sequences::uniform_sequence<float, 0> inner_seq
        (inner_length,
         outer_length,
         thrust::raw_pointer_cast(&y[0]));
    cooperative::sequences::uniform_sequence<float, 1> outer_seq
        (outer_length,
         1,
         inner_seq);

    thrust::device_vector<float> r(outer_length);

    group_sum<cooperative::sequences::uniform_sequence<float, 0> > f;
    
    thrust::transform(thrust::retag<cooperative::cuda::tag>(outer_seq.begin()),
                      thrust::retag<cooperative::cuda::tag>(outer_seq.end()),
                      thrust::retag<cooperative::cuda::tag>(r.begin()),
                      f);


    thrust::copy(r.begin(), r.end(), std::ostream_iterator<float>(std::cout, "\n"));                   
    
}
