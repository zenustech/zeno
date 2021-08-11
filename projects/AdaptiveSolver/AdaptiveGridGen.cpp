#include "AdaptiveGridGen.h"
#include <tbb/parallel_for.h>
namespace zeno{
void AdaptiveIndexGenerator::generateAdaptiveGrid(
        AdaptiveIndexGenerator& data, 
        int max_levels, 
        double target_h,
        zinc::vec3f &bmin,
        zinc::vec3f &bmax
        /*I think better to use a zeno::functionObj here*/
        )
{
    data.IndexLevels.resize(max_levels);
    std::vector<double> h;
    for(int i=0; i<max_levels; i++)
    {
        h.push_back(target_h * std::pow(2.0,(double)i));
    }

    //we shall assume level_max is already provided, by
    //particular method
    for(int i = max_levels-2; i>=0; i--)
    {
        //loop over voxels of coarser level
        tbb::parallel_for(
            (size_t)0,
            (size_t)data.IndexLevels[i+1].size(),
            (size_t)1,
            [&](size_t index)
            {
                //get voxel
                zinc::vec3i voxelic = data.IndexLevels[i+1][index];
                zinc::vec3f voxel_corner = 
                zinc::vec3f(voxelic)*h[i+1]; 
                bool to_emit;
                /*let the function Obj decide if emit*/
                if(to_emit){
                    data.IndexLevels[i].emplace_back(voxelic*2); 
                    data.IndexLevels[i].emplace_back(voxelic*2+ zinc::vec3i(1,0,0)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(0,1,0)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(1,1,0)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(0,0,1)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(1,0,1)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(0,1,1)); 
                    data.IndexLevels[i].emplace_back(voxelic*2+zinc::vec3i(1,1,1)); 
                }
            }
        );
        data.hLevels.push_back(h[i]);
    }
}
}