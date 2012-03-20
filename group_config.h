#pragma once

namespace cooperative {

struct group_config {
    int group_size;
    int scratch_size;
    __host__ __device__
    group_config(int g, int s): group_size(g), scratch_size(s) {}
};

}
