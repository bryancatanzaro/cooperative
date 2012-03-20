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
#include <thrust/iterator/retag.h>
#include <cooperative/cuda/tag.h>
#include <cooperative/is_cooperative.h>
#include <cooperative/cuda/functional.h>
#include <cooperative/cuda/for_each.h>

namespace cooperative {
namespace cuda {

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
OutputIterator
transform(tag,
          InputIterator first,
          InputIterator last,
          OutputIterator result,
          UnaryFunction op) {
    typedef thrust::tuple<InputIterator, OutputIterator> IteratorTuple;
    typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

    ZipIterator zipped_result =
        cooperative::cuda::for_each(tag(),
                                    thrust::make_zip_iterator(thrust::make_tuple(first, result)),
                                    thrust::make_zip_iterator(thrust::make_tuple(last, result)),
                                    unary_cooperative_transform_functor<UnaryFunction>(op));
    return thrust::get<1>(zipped_result.get_iterator_tuple());
                         
}


}
}
