/*
 *   Copyright (c) 2012, NVIDIA CORPORATION.  All rights reserved.
 * 
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 * 
 *       http://www.apache.org/licenses/LICENSE-2.0
 * 
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * 
 */

#pragma once
#include <cooperative/group_config.h>
#include <cooperative/cuda/thread_group.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>

namespace cooperative {
namespace cuda {

template<typename Argument,
         typename Result>
struct unary_cooperative_function
    : thrust::unary_function<Argument, Result> {
    
    thread_group group;
    cooperative::group_config c;
    __host__ __device__
    unary_cooperative_function(const cooperative::group_config &_c)
        : c(_c) {}
    
    __device__
    void set_group(const thread_group& in) {
        group = in;
    }

    __host__ __device__
    cooperative::group_config get_config() const {
        return c;
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

    __host__ __device__
    cooperative::group_config get_config() const {
        return f.get_config();
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
