#pragma once

#include <tbb/concurrent_vector.h>
#include <vector>
#include <zeno/zeno.h>
#include "tbb/scalable_allocator.h"
#include <zeno/ZenoInc.h>
namespace zeno{
struct AdaptiveIndexGenerator{
    std::vector<tbb::concurrent_vector<zinc::vec3i>> IndexLevels;
    std::vector<double> hLevels;
    void generateAdaptiveGrid(
        AdaptiveIndexGenerator& data, 
        int max_levels, 
        double target_h,
        zinc::vec3f &bmin,
        zinc::vec3f &bmax
        /*I think better to use a zeno::functionObj here*/
        );

};

}