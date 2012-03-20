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
