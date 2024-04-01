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
        int j = tidx/nx;
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
void edgeLoopSum(zeno::PrimitiveObject* prim, int nx, int ny, std::string &channel, std::string addChannel)
{
    auto &l = prim->attr<T>("l"+channel);
    auto &r = prim->attr<T>("r"+channel);
    auto &t = prim->attr<T>("t"+channel);
    auto &b = prim->attr<T>("b"+channel);
    auto &res = prim->attr<T>(addChannel);
#pragma omp parallel for
    for(int tidx = 0; tidx<nx*ny; tidx++)
    {
        int i = tidx%nx;
        int j = tidx/nx;
        size_t lidx = j * nx + max(i-1,0);
        size_t ridx = j * nx + min(i+1, nx-1);
        size_t uidx = min(j+1, ny-1) * nx  + i;
        size_t bidx = max(j-1, 0) * nx + i;

        res[tidx] += r[lidx] + l[ridx] + t[bidx] + b[tidx];
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
    for(auto tidx = 0; tidx<nx*ny; tidx++)
    {
        int i = tidx%nx;
        int j = tidx/nx;
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
template<class T>
void cornerLoopSum(zeno::PrimitiveObject* prim, int nx, int ny, std::string &channel, std::string addChannel)
{
    auto &lt  = prim->attr<T>("lt"+channel);
    auto &rt  = prim->attr<T>("rt"+channel);
    auto &lb  = prim->attr<T>("lb"+channel);
    auto &rb  = prim->attr<T>("rb"+channel);
    auto &res = prim->attr<T>(addChannel);
#pragma omp parallel for
    for(auto tidx = 0; tidx<nx*ny; tidx++)
    {
        int i = tidx%nx;
        int j = tidx/nx;
        size_t ltidx = min(j+1, ny-1) * nx + max(i-1,0);
        size_t rtidx = min(j+1, ny-1) * nx + min(i+1, nx-1);
        size_t lbidx = max(j-1, 0) * nx  + max(i-1,0);
        size_t rbidx = max(j-1, 0) * nx + min(i+1, nx-1);

        res[tidx] += lt[rbidx] + rt[lbidx] + lb[rtidx] + rb[ltidx];
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

struct MomentumTransfer2DFiniteDifference : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto grid = get_input<zeno::PrimitiveObject>("grid");
        auto attrT = get_param<std::string>("attrT");
        auto type = get_param<std::string>("OpType");
        auto channel = get_input<zeno::StringObject>("channel")->get();
        auto addChannel = get_input<zeno::StringObject>("add_channel")->get();
        if(attrT=="float")
        {
            grid->add_attr<float>(addChannel);
        }
        if(attrT=="vec3")
        {
            grid->add_attr<zeno::vec3f>(addChannel);
        }

        if(grid->has_attr(channel) && grid->has_attr(addChannel))
        {
            if(type == "FIVE_STENCIL" || type == "NINE_STENCIL")
            {
                if(attrT == "float")
                {
                    edgeLoopSum<float>(grid.get(), nx, ny, channel, addChannel);
                }
                if(attrT == "vec3")
                {
                    edgeLoopSum<zeno::vec3f>(grid.get(), nx, ny, channel, addChannel);
                }
            }
            if(type == "NINE_STENCIL")
            {
                if(attrT == "float")
                {
                    cornerLoopSum<float>(grid.get(), nx, ny, channel, addChannel);
                }
                if(attrT == "vec3")
                {
                    cornerLoopSum<zeno::vec3f>(grid.get(), nx, ny, channel, addChannel);
                }
            }
        }

        set_output("prim", std::move(grid));

    }
};
ZENDEFNODE(MomentumTransfer2DFiniteDifference, {
                                         {{"PrimitiveObject", "grid"},
                                          {"int", "nx","1"},
                                          {"int", "ny", "1"},
                                          {"string", "channel", "d"},
                                          {"string", "add_channel", "d"}},
                                         {{"PrimitiveObject", "prim"}},
                                         {{"enum vec3 float", "attrT", "float"},
                                          {"enum FIVE_STENCIL NINE_STENCIL", "OpType", "FIVE_STENCIL"}},
                                         {"zenofx"},
                                     });

template <class T>
T lerp(T a, T b, float c) {
    return (1.0 - c) * a + c * b;
}
template <class T>
void sample2D(std::vector<zeno::vec3f> &coord, std::vector<T> &field, std::vector<T> &primAttr, int nx, int ny, float h,
              zeno::vec3f bmin, bool isPeriodic) {
    std::vector<T> temp(coord.size());
#pragma omp parallel for
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
        auto uv = coord[tidx];
        zeno::vec3f uv2;
        if (isPeriodic) {
            auto Lx = (nx - 1) * h;
            auto Ly = (ny - 1) * h;
            int gid_x = std::floor( ( uv[0] - bmin[0] ) / Lx );
            int gid_y = std::floor( ( uv[2] - bmin[2] ) / Ly );
            uv2 = ( uv - (bmin + zeno::vec3f{gid_x * Lx, 0, gid_y * Ly}) ) / h;
            uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.00, 0.0, 0.00)),
                            zeno::vec3f((float)nx - 1.01, 0.0, (float)ny - 1.01));
        } else {
            uv2 = (uv - bmin) / h;
            uv2 = zeno::min(zeno::max(uv2, zeno::vec3f(0.01, 0.0, 0.01)),
                            zeno::vec3f((float)nx - 1.01, 0.0, (float)ny - 1.01));
        }
        
        int i = uv2[0];
        int j = uv2[2];
        float cx = uv2[0] - i, cy = uv2[2] - j;
        size_t idx00 = j * nx + i, idx01 = j * nx + i + 1, idx10 = (j + 1) * nx + i, idx11 = (j + 1) * nx + i + 1;
        temp[tidx] = lerp<T>(lerp<T>(field[idx00], field[idx01], cx), lerp<T>(field[idx10], field[idx11], cx), cy);
    }
#pragma omp parallel for
    for (auto tidx = 0; tidx < coord.size(); tidx++) {
        primAttr[tidx] = temp[tidx];
    }
}
struct Grid2DSample : zeno::INode {
    virtual void apply() override {
        auto nx = get_input<zeno::NumericObject>("nx")->get<int>();
        auto ny = get_input<zeno::NumericObject>("ny")->get<int>();
        auto bmin = get_input2<zeno::vec3f>("bmin");
        auto prim = get_input<zeno::PrimitiveObject>("prim");
        auto grid = get_input<zeno::PrimitiveObject>("sampleGrid");
        auto channelList = get_input2<std::string>("channel");
        auto sampleby = get_input2<std::string>("sampleBy");
        auto isPeriodic = get_input2<std::string>("sampleType") == "Periodic";
        auto h = get_input<zeno::NumericObject>("h")->get<float>();

        bool sampleAll = false;

        std::vector<std::string> channels;
        std::istringstream iss(channelList);
        std::string word;
        while (iss >> word) {
            if (word == "*") {
                sampleAll = true;
                channels = grid->attr_keys();
                break;
            }
            channels.push_back(word);
        }

        if (prim->has_attr(sampleby)) {
            if (!(sampleby == "pos" || std::holds_alternative<std::vector<vec3f>>(prim->attr(sampleby))))
                throw std::runtime_error("[sampleBy] has to be a vec3f attribute!");

            for (const auto &ch : channels) {
                if (ch != "pos") {
                    std::visit(
                        [&](auto &&ref) {
                            using T = std::remove_cv_t<std::remove_reference_t<decltype(ref[0])>>;
                            prim->add_attr<T>(ch);
                            sample2D<T>(prim->attr<zeno::vec3f>(sampleby), grid->attr<T>(ch), prim->attr<T>(ch), nx, ny,
                                        h, bmin, isPeriodic);
                        },
                        grid->attr(ch));
                }
            }
        }

        set_output("prim", std::move(prim));
    }
};
ZENDEFNODE(Grid2DSample, {
                             {{"PrimitiveObject", "prim"},
                              {"PrimitiveObject", "sampleGrid"},
                              {"int", "nx", "1"},
                              {"int", "ny", "1"},
                              {"float", "h", "1"},
                              {"vec3f", "bmin", "0,0,0"},
                              {"string", "channel", "*"},
                              {"string", "sampleBy", "pos"},
                              {"enum Clamp Periodic", "sampleType", "Clamp"}},
                             {{"PrimitiveObject", "prim"}},
                             {},
                             {"zenofx"},
                         });
} // namespace zeno