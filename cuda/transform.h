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
typename thrust::detail::enable_if<is_cooperative<UnaryFunction>::value,
                                   OutputIterator>::type
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

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
typename thrust::detail::disable_if<is_cooperative<UnaryFunction>::value,
                                    OutputIterator>::type
transform(tag,
          InputIterator first,
          InputIterator last,
          OutputIterator result,
          UnaryFunction op) {
    thrust::transform(thrust::retag<thrust::system::cuda::tag>(first),
                      thrust::retag<thrust::system::cuda::tag>(last),
                      thrust::retag<thrust::system::cuda::tag>(result),
                      op);
}

}
}
