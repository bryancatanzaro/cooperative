#include <cooperative/cuda/tag.h>
#include <cooperative/cuda/functional.h>
#include <cooperative/cuda/transform.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>
#include <cooperative/group_config.h>
#include <cooperative/sequences/uniform_sequence.h>
#include <cooperative/sequences/make_uniform_sequence.h>
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
        //Simple block parallel sum
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


using namespace cooperative::sequences;

size_t round_up_to_multiple(size_t d, size_t m) {
    return (((d-1)/m)+1)*m;
}

void initialize_data(uniform_sequence<float, 1> h_seq) {
    for(size_t i = 0; i < h_seq.size(); i++) {
        for(size_t j = 0; j < h_seq[i].size(); j++) {
            if (j==0) {
                h_seq[i][j] = i;
            } else {
                h_seq[i][j] = 0.1;
            }
        }
    }
}

void sequential_test(uniform_sequence<float, 1> h_seq,
                     thrust::host_vector<float>& o) {
    for(size_t i = 0; i < h_seq.size(); i++) {
        float accumulator = 0;
        for(size_t j = 0; j < h_seq[i].size(); j++) {
            accumulator += h_seq[i][j];
        }
        o[i] = accumulator;
    }
}

float error(thrust::host_vector<float>& g,
            thrust::device_vector<float>& r) {
    thrust::host_vector<float> h = r;
    float accumulator = 0;
    for(size_t i = 0; i < g.size(); i++) {
        float delta = g[i] - h[i];
        accumulator += delta * delta;
    }
    return accumulator;
}

int main() {
    //Lengths
    size_t outer_length = 10;
    size_t inner_length = 32;
    
    //Padded, column-major data layout is determined by strides
    size_t outer_stride = 1;
    size_t inner_stride = round_up_to_multiple(outer_length, 16);

    size_t storage_length = inner_stride * inner_length;

    //Allocate container
    thrust::host_vector<float> x(storage_length);

    //Make uniform nested sequence view
    uniform_sequence<float, 1> h_seq =
        make_uniform_sequence(
            thrust::make_tuple(outer_length, inner_length),
            thrust::make_tuple(outer_stride, inner_stride),
            thrust::raw_pointer_cast(&x[0]));

    //Fill in some test data
    initialize_data(h_seq);

    //Compute sequential result
    thrust::host_vector<float> golden(outer_length);
    sequential_test(h_seq, golden);


    //Make uniform nested sequence view on device
    thrust::device_vector<float> y = x;

    uniform_sequence<float, 1> seq =
        make_uniform_sequence(
            thrust::make_tuple(outer_length, inner_length),
            thrust::make_tuple(outer_stride, inner_stride),
            thrust::raw_pointer_cast(&y[0]));

    thrust::device_vector<float> r(outer_length);

    group_sum<uniform_sequence<float, 0> > f;

    //Launch cooperative transform
    thrust::transform(thrust::retag<cooperative::cuda::tag>(seq.begin()),
                      thrust::retag<cooperative::cuda::tag>(seq.end()),
                      thrust::retag<cooperative::cuda::tag>(r.begin()),
                      f);

    //Check errors
    std::cout << "Error: " << error(golden, r) << std::endl;
    
}
