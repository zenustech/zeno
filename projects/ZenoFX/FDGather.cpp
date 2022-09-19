//
// copyright zenustech
//
#include <zeno/zeno.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/DictObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/core/Graph.h>
#include <zfx/zfx.h>
#include <zfx/x64.h>
#include <cassert>
#include "dbg_printf.h"

namespace zeno {
template<class T>
void edgeLoop(zeno::PrimitiveObject* prim, int nx, int ny, std::string &channel)
{
    auto &l = prim->add_attr<T>("l"+channel);
    auto &r = prim->add_attr<T>("r"+channel);
    auto &t = prim->add_attr<T>("t"+channel);
    auto &b = prim->add_attr<T>("b"+channel);
#pragma omp parallel for
    for(int tidx = 0; tidx<nx*ny; tidx++)
    {
        int i = tidx%nx;
        int j = tidx/ny;
        size_t lidx = j * nx + max(i-1,0);
        size_t ridx = j * nx + min(i+1, nx-1);
        size_t uidx = min(j+1, ny-1) * nx  + i;
        size_t bidx = max(j-1, 0) * nx + i;

        l[tidx] = prim->attr<T>(channel)[lidx];
        r[tidx] = prim->attr<T>(channel)[ridx];
        t[tidx] = prim->attr<T>(channel)[uidx];
        b[tidx] = prim->attr<T>(channel)[bidx];
    }
}
template<class T>
void cornerLoop(zeno::PrimitiveObject* prim, int nx, int ny, std::string &channel)
{
    auto &lt = prim->add_attr<T>("lt"+channel);
    auto &rt = prim->add_attr<T>("rt"+channel);
    auto &lb = prim->add_attr<T>("lb"+channel);
    auto &rb = prim->add_attr<T>("rb"+channel);
#pragma omp parallel for
    for(size_t tidx = 0; tidx<nx*ny; tidx++)
    {
        int i = tidx%nx;
        int j = tidx/ny;
        size_t ltidx = min(j+1, ny-1) * nx + max(i-1,0);
        size_t rtidx = min(j+1, ny-1) * nx + min(i+1, nx-1);
        size_t lbidx = max(j-1, 0) * nx  + max(i-1,0);
        size_t rbidx = max(j-1, 0) * nx + min(i+1, nx-1);

        lt[tidx] = prim->attr<T>(channel)[ltidx];
        rt[tidx] = prim->attr<T>(channel)[rtidx];
        lb[tidx] = prim->attr<T>(channel)[lbidx];
        rb[tidx] = prim->attr<T>(channel)[rbidx];
    }
}
struct Gather2DFiniteDifference : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto grid = get_input<zeno::PrimitiveObject>("grid");
        auto attrT = get_param<std::string>("attrT");
        auto type = get_param<std::string>("OpType");
        auto channel = get_input<zeno::StringObject>("channel")->get();

        if(grid->has_attr(channel))
        {
            if(type == "FIVE_STENCIL" || type == "NINE_STENCIL")
            {
                if(attrT == "float")
                {
                    edgeLoop<float>(grid.get(), nx, ny, channel);
                }
                if(attrT == "vec3")
                {
                    edgeLoop<zeno::vec3f>(grid.get(), nx, ny, channel);
                }
            }
            if(type == "NINE_STENCIL")
            {
                if(attrT == "float")
                {
                    cornerLoop<float>(grid.get(), nx, ny, channel);
                }
                if(attrT == "vec3")
                {
                    cornerLoop<zeno::vec3f>(grid.get(), nx, ny, channel);
                }
            }
        }

        set_output("prim", std::move(grid));

    }
};

ZENDEFNODE(Gather2DFiniteDifference, {
                                         {{"PrimitiveObject", "grid"},
                                          {"int", "nx","1"},
                                          {"int", "ny", "1"},
                                          {"string", "channel", "pos"}},
                                         {{"PrimitiveObject", "prim"}},
                                         {{"enum vec3 float", "attrT", "float"},
                                          {"enum FIVE_STENCIL NINE_STENCIL", "OpType", "FIVE_STENCIL"}},
                                         {"zenofx"},
                                     });

}