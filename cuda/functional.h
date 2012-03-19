#pragma once

#include <cooperative/cuda/thread_group.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace cooperative {
namespace cuda {

template<typename Argument,
         typename Result>
struct unary_cooperative_function
    : thrust::unary_function<Argument, Result> {
    thread_group g;
    
    __device__
    void set_group(const thread_group& in) {
        g = in;
    }
};


template<typename UnaryCooperativeFunction>
struct unary_cooperative_transform_functor {
    typedef void result_type;
    UnaryCooperativeFunction f;
    thread_group g;
    
    unary_cooperative_transform_functor(UnaryCooperativeFunction _f)
        : f(_f) {}

    __device__
    void set_group(const thread_group& in) {
        g = in;
        f.set_group(g);
    }
    
    template<typename Tuple>
    __device__
    inline result_type operator()(Tuple t) {
        //Group cooperatively computes result
        typename UnaryCooperativeFunction::result_type r = f(thrust::get<0>(t));
        //Thread 0 stores result
        if (g.local_thread_id() == 0) {
            thrust::get<1>(t) = r;
        }
    }
};



}
}
