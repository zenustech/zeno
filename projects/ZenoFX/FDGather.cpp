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
template<class T>
T lerp(T a, T b, float c)
{
    return (1.0 - c) * a + c * b;
}
template<class T>
void sample2D(std::vector<zeno::vec3f> &coord, std::vector<T> &field, int nx, int ny, float h, zeno::vec3f bmin)
{
    std::vector<T> temp(field.size());
#pragma omp parallel for
    for(size_t tidx=0;tidx<coord.size();tidx++)
    {
        auto uv = coord[tidx];
        auto uv2 = (uv - bmin) / h;
        uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.01,0.0,0.01)), zeno::vec3f(nx-1.01, 0.0, ny-1.01));
        int i = uv2[0];
        int j = uv2[2];
        float cx = uv2[0] - i, cy = uv2[2] - j;
        size_t idx00 = j*nx + i, idx01 = j*nx + i + 1, idx10 = (j+1)*nx + i, idx11 = (j+1)*nx + i + 1;
        temp[tidx] = lerp<T>(lerp<T>(field[idx00], field[idx01], cx), lerp<T>(field[idx10], field[idx11], cx), cy);
    }
#pragma omp parallel for
    for(size_t tidx=0;tidx<coord.size();tidx++)
    {
        field[tidx]=temp[tidx];
    }
}
struct Grid2DSample : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto bmin = get_input2<zeno::vec3f>("bmin");
        auto grid = get_input<zeno::PrimitiveObject>("grid");
        auto attrT = get_param<std::string>("attrT");
        auto channel = get_input<zeno::StringObject>("channel")->get();
        auto sampleby = get_input<zeno::StringObject>("sampleBy")->get();
        auto h = get_input<zeno::NumericObject>("h")->get<float>();
        if(grid->has_attr(channel) && grid->has_attr(sampleby))
        {
            if(attrT == "float")
            {
                sample2D<float>(grid->attr<zeno::vec3f>(sampleby), grid->attr<float>(channel), nx, ny, h, bmin);
            }
            else if(attrT == "vec3f")
            {
                sample2D<zeno::vec3f>(grid->attr<zeno::vec3f>(sampleby), grid->attr<zeno::vec3f>(channel), nx, ny, h, bmin);
            }
        }

        set_output("prim", std::move(grid));

    }
};
ZENDEFNODE(Grid2DSample, {
                                         {{"PrimitiveObject", "grid"},
                                          {"int", "nx","1"},
                                          {"int", "ny", "1"},
                                          {"float", "h", "1"},
                                          {"vec3f","bmin", "0,0,0" },
                                          {"string", "channel", "pos"},
                                          {"string", "sampleBy", "pos"}},
                                         {{"PrimitiveObject", "prim"}},
                                         {{"enum vec3 float", "attrT", "float"},
                                          },
                                         {"zenofx"},
                                     });
}