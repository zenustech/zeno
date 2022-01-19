#include "GridData.cuh"

namespace zeno
{
    void GridData::initData(vec3f bmin, vec3f bmax, float dx, float dt)
    {
        parm.bmin = bmin;
        parm.dx = dx;
        parm.dt = dt;
        parm.gridNum = ceil((bmax-bmin) / dx);
        for(int i=0;i<3;++i)
            if(parm.gridNum[i] <= 0)
                parm.gridNum[i] = 1;
        parm.bmax = bmin + parm.gridNum * dx;
        //std::cout<<"grid Nun is (" << parm.gridNum[0]<<", "<<parm.gridNum[1]<<", "<<parm.gridNum[2]<<std::endl;
        // printf("grid Num is (%d,%d,%d)\n", parm.gridNum[0], parm.gridNum[1], parm.gridNum[2]);
    }

    void GridData::computeKey(float bmin[3], float bmax[3])
    {
        
    }

    struct generateAdaptiveGridGPU : zeno::INode{
        virtual void apply() override {
            auto bmin = get_input("bmin")->as<zeno::NumericObject>()->get<vec3f>();
            auto bmax = get_input("bmax")->as<zeno::NumericObject>()->get<vec3f>();
            auto dx = get_input("dx")->as<zeno::NumericObject>()->get<float>();
            float dt = get_input("dt")->as<zeno::NumericObject>()->get<float>();
            //int levelNum = get_input("levelNum")->as<zeno::NumericObject>()->get<int>();
                
            auto data = zeno::IObject::make<GridData>();
            data->initData(bmin, bmax, dx, dt);
            set_output("gridData", data);
        }
    };
    ZENDEFNODE(generateAdaptiveGridGPU, {
            {"bmin", "bmax", "dx", "dt"},
            {"gridData"},
            {},
            {"AdaptiveSolver"},
    });
}