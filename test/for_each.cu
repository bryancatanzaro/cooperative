#include <cooperative/cuda/tag.h>
#include <cooperative/cuda/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/device_vector.h>

int main() {
    thrust::counting_iterator<int> x;
    int length = 10;
    thrust::device_vector<int> y(length);
    thrust::transform(thrust::retag<cooperative::cuda::tag>(x),
                      thrust::retag<cooperative::cuda::tag>(x+length),
                      thrust::retag<cooperative::cuda::tag>(y.begin()),
                      thrust::identity<int>());
    
}
