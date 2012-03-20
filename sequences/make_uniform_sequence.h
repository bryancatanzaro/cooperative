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
#include <thrust/tuple.h>
#include <cooperative/sequences/uniform_sequence.h>

namespace cooperative {
namespace sequences {

template<typename T, int D, typename Tuple>
struct make_uniform_sequence_impl {
    static uniform_sequence<T, D> make(
        const Tuple& l, const Tuple& s, T* d, size_t o) {
        return uniform_sequence<T, D>(
            l.head,
            s.head,
            make_uniform_sequence_impl<T,
                                       D-1,
                                       typename Tuple::tail_type>::make(
                                           l.tail,
                                           s.tail,
                                           d,
                                           o));
    }
};

template<typename T, typename Tuple>
struct make_uniform_sequence_impl<T,
                                  0,
                                  Tuple > {
    static const int D = 0;
    static uniform_sequence<T, D> make(
        const Tuple& l, const Tuple& s, T* d, size_t o) {
        return uniform_sequence<T, D>(l.head, s.head, d, o);
    }
};

template<typename T, typename Tuple>
uniform_sequence<T, thrust::tuple_size<Tuple>::value - 1>
make_uniform_sequence(const Tuple& l,
                      const Tuple& s,
                      T* d,
                      size_t o=0) {
    return make_uniform_sequence_impl<T,
                                      thrust::tuple_size<Tuple>::value-1,
                                      Tuple>::make(l, s, d, o);
}

}
}
