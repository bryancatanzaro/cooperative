#pragma once

namespace cooperative {
namespace cuda {

class thread_group {
private:
    void* m_scratch;
    unsigned int m_scratch_size;
public:
    __host__ __device__
    thread_group(void* scratch, unsigned int scratch_size)
        : m_scratch(scratch), m_scratch_size(scratch_size) {}
    __host__ __device__
    thread_group() : m_scratch(NULL), m_scratch_size(0) {}

    __device__
    unsigned int scratch_size() const {
        return m_scratch_size;
    }

    template<typename T>
    __device__
    T* scratch() const {
        return reinterpret_cast<T*>(m_scratch);
    }


    __device__
    void barrier() const {
        __syncthreads();
    }

    __device__
    int global_thread_id() const {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__
    int local_thread_id() const {
        return threadIdx.x;
    }

    __device__
    int group_id() const {
        return blockIdx.x;
    }

    __device__
    int group_size() const {
        return blockDim.x;
    }

    __device__
    int number_of_groups() const {
        return gridDim.x;
    }

};

}
}
