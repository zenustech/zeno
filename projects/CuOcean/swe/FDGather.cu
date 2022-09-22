#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/types/Property.h"
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>

namespace zeno {

template <int nchn>
void edgeLoop(typename ZenoParticles::particles_t &prim, int nx, int ny, const std::string &channel) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto pol = cuda_exec().device(0);
    const SmallString lTag = std::string("l") + channel;
    const SmallString rTag = std::string("r") + channel;
    const SmallString tTag = std::string("t") + channel;
    const SmallString bTag = std::string("b") + channel;
    const SmallString tag = channel;
    prim.append_channels(
        pol, {{lTag.asString(), nchn}, {rTag.asString(), nchn}, {tTag.asString(), nchn}, {bTag.asString(), nchn}});
    pol(Collapse{nx, ny},
        [verts = proxy<space>({}, prim), nx, ny, lTag, rTag, tTag, bTag, tag] ZS_LAMBDA(int i, int j) mutable {
            size_t lidx = j * nx + math::max(i - 1, 0);
            size_t ridx = j * nx + math::min(i + 1, nx - 1);
            size_t tidx = math::min(j + 1, ny - 1) * nx + i;
            size_t bidx = math::max(j - 1, 0) * nx + i;
            size_t idx = j * nx + i;

            verts.template tuple<nchn>(lTag, idx) = verts.template pack<nchn>(tag, lidx);
            verts.template tuple<nchn>(rTag, idx) = verts.template pack<nchn>(tag, ridx);
            verts.template tuple<nchn>(tTag, idx) = verts.template pack<nchn>(tag, tidx);
            verts.template tuple<nchn>(bTag, idx) = verts.template pack<nchn>(tag, bidx);
        });
}
template <int nchn>
void edgeLoopSum(typename ZenoParticles::particles_t &prim, int nx, int ny, const std::string &channel,
                 const std::string &addChannel) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto pol = cuda_exec().device(0);
    const SmallString lTag = std::string("l") + channel;
    const SmallString rTag = std::string("r") + channel;
    const SmallString tTag = std::string("t") + channel;
    const SmallString bTag = std::string("b") + channel;
    const SmallString tag = addChannel;
    pol(Collapse{nx, ny},
        [verts = proxy<space>({}, prim), nx, ny, lTag, rTag, tTag, bTag, tag] ZS_LAMBDA(int i, int j) mutable {
            size_t lidx = j * nx + math::max(i - 1, 0);
            size_t ridx = j * nx + math::min(i + 1, nx - 1);
            size_t tidx = math::min(j + 1, ny - 1) * nx + i;
            size_t bidx = math::max(j - 1, 0) * nx + i;
            size_t idx = j * nx + i;

            verts.template tuple<nchn>(tag, idx) =
                verts.template pack<nchn>(rTag, lidx) + verts.template pack<nchn>(lTag, ridx) +
                verts.template pack<nchn>(tTag, bidx) + verts.template pack<nchn>(bTag, tidx);
        });
}

template <int nchn>
void cornerLoop(typename ZenoParticles::particles_t &prim, int nx, int ny, const std::string &channel) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto pol = cuda_exec().device(0);
    const SmallString ltTag = std::string("lt") + channel;
    const SmallString rtTag = std::string("rt") + channel;
    const SmallString lbTag = std::string("lb") + channel;
    const SmallString rbTag = std::string("rb") + channel;
    const SmallString tag = channel;
    prim.append_channels(
        pol, {{ltTag.asString(), nchn}, {rtTag.asString(), nchn}, {lbTag.asString(), nchn}, {rbTag.asString(), nchn}});
    pol(Collapse{nx, ny},
        [verts = proxy<space>({}, prim), nx, ny, ltTag, rtTag, lbTag, rbTag, tag] ZS_LAMBDA(int i, int j) mutable {
            size_t ltidx = math::min(j + 1, ny - 1) * nx + math::max(i - 1, 0);
            size_t rtidx = math::min(j + 1, ny - 1) * nx + math::min(i + 1, nx - 1);
            size_t lbidx = math::max(j - 1, 0) * nx + math::max(i - 1, 0);
            size_t rbidx = math::max(j - 1, 0) * nx + math::min(i + 1, nx - 1);
            size_t idx = j * nx + i;

            verts.template tuple<nchn>(ltTag, idx) = verts.template pack<nchn>(tag, ltidx);
            verts.template tuple<nchn>(rtTag, idx) = verts.template pack<nchn>(tag, rtidx);
            verts.template tuple<nchn>(lbTag, idx) = verts.template pack<nchn>(tag, lbidx);
            verts.template tuple<nchn>(rbTag, idx) = verts.template pack<nchn>(tag, rbidx);
        });
}
template <int nchn>
void cornerLoopSum(typename ZenoParticles::particles_t &prim, int nx, int ny, const std::string &channel,
                   const std::string &addChannel) {
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto pol = cuda_exec().device(0);
    const SmallString ltTag = std::string("lt") + channel;
    const SmallString rtTag = std::string("rt") + channel;
    const SmallString lbTag = std::string("lb") + channel;
    const SmallString rbTag = std::string("rb") + channel;
    const SmallString tag = addChannel;
    pol(Collapse{nx, ny},
        [verts = proxy<space>({}, prim), nx, ny, ltTag, rtTag, lbTag, rbTag, tag] ZS_LAMBDA(int i, int j) mutable {
            size_t ltidx = math::min(j + 1, ny - 1) * nx + math::max(i - 1, 0);
            size_t rtidx = math::min(j + 1, ny - 1) * nx + math::min(i + 1, nx - 1);
            size_t lbidx = math::max(j - 1, 0) * nx + math::max(i - 1, 0);
            size_t rbidx = math::max(j - 1, 0) * nx + math::min(i + 1, nx - 1);
            size_t idx = j * nx + i;

            verts.template tuple<nchn>(tag, idx) =
                verts.template pack<nchn>(ltTag, rbidx) + verts.template pack<nchn>(rtTag, lbidx) +
                verts.template pack<nchn>(lbTag, rtidx) + verts.template pack<nchn>(rbTag, ltidx);
        });
}
struct ZSGather2DFiniteDifference : zeno::INode {
    virtual void apply() override {
        auto nx = get_input2<int>("nx");
        auto ny = get_input2<int>("ny");
        auto grid = get_input<ZenoParticles>("grid");
        auto attrT = get_input2<std::string>("attrT");
        auto type = get_input2<std::string>("OpType");
        auto channel = get_input2<std::string>("channel");

        if (auto &verts = grid->getParticles(); verts.hasProperty(channel)) {
            if (type == "FIVE_STENCIL" || type == "NINE_STENCIL") {
                if (attrT == "float") {
                    edgeLoop<1>(verts, nx, ny, channel);
                }
                if (attrT == "vec3") {
                    edgeLoop<3>(verts, nx, ny, channel);
                }
            }
            if (type == "NINE_STENCIL") {
                if (attrT == "float") {
                    cornerLoop<1>(verts, nx, ny, channel);
                }
                if (attrT == "vec3") {
                    cornerLoop<3>(verts, nx, ny, channel);
                }
            }
        }

        set_output("prim", std::move(grid));
    }
};

ZENDEFNODE(ZSGather2DFiniteDifference, {
                                           {{"ZSParticles", "grid"},
                                            {"int", "nx", "1"},
                                            {"int", "ny", "1"},
                                            {"string", "channel", "pos"},
                                            {"enum vec3 float", "attrT", "float"},
                                            {"enum FIVE_STENCIL NINE_STENCIL", "OpType", "FIVE_STENCIL"}},
                                           {{"ZSParticles", "prim"}},
                                           {},
                                           {"zenofx"},
                                       });

struct ZSMomentumTransfer2DFiniteDifference : zeno::INode {
    void apply() override {
        auto nx = get_input2<int>("nx");
        auto ny = get_input2<int>("ny");
        auto grid = get_input<ZenoParticles>("grid");
        auto attrT = get_input2<std::string>("attrT");
        auto type = get_input2<std::string>("OpType");
        auto channel = get_input2<std::string>("channel");
        auto addChannel = get_input2<std::string>("add_channel");

        auto &verts = grid->getParticles();
        auto pol = zs::cuda_exec().device(0);
        if (attrT == "float") {
            verts.append_channels(pol, {{addChannel, 1}});
        }
        if (attrT == "vec3") {
            verts.append_channels(pol, {{addChannel, 3}});
        }

        if (verts.hasProperty(channel) && verts.hasProperty(addChannel)) {
            if (type == "FIVE_STENCIL" || type == "NINE_STENCIL") {
                if (attrT == "float") {
                    edgeLoopSum<1>(verts, nx, ny, channel, addChannel);
                }
                if (attrT == "vec3") {
                    edgeLoopSum<3>(verts, nx, ny, channel, addChannel);
                }
            }
            if (type == "NINE_STENCIL") {
                if (attrT == "float") {
                    cornerLoopSum<1>(verts, nx, ny, channel, addChannel);
                }
                if (attrT == "vec3") {
                    cornerLoopSum<3>(verts, nx, ny, channel, addChannel);
                }
            }
        }

        set_output("prim", std::move(grid));
    }
};

ZENDEFNODE(ZSMomentumTransfer2DFiniteDifference, {
                                                     {{"ZenoParticles", "grid"},
                                                      {"int", "nx", "1"},
                                                      {"int", "ny", "1"},
                                                      {"string", "channel", "d"},
                                                      {"string", "add_channel", "d"},
                                                      {"enum vec3 float", "attrT", "float"},
                                                      {"enum FIVE_STENCIL NINE_STENCIL", "OpType", "FIVE_STENCIL"}},
                                                     {{"ZenoParticles", "prim"}},
                                                     {},
                                                     {"zenofx"},
                                                 });

template <class T> static constexpr T lerp(T a, T b, float c) {
    return (1 - c) * a + c * b;
}
template <auto nchn>
void sample2D(typename ZenoParticles::particles_t &prim, const zs::SmallString &coordTag,
              const zs::SmallString &fieldTag, int nx, int ny, float h, zs::vec<float, 3> bmin) {
    using vec3f = zs::vec<float, 3>;
    using namespace zs;
    constexpr auto space = execspace_e::cuda;
    auto pol = cuda_exec().device(0);
    using T = conditional_t<nchn == 1, zs::vec<float, 1>, vec3f>;

    Vector<T> temp(prim.get_allocator(), prim.size());

    pol(range(prim.size()), [temp = proxy<space>(temp), prim = proxy<space>({}, prim), coordTag, fieldTag, bmin, nx, ny,
                             h] ZS_LAMBDA(int tidx) mutable {
        auto uv = prim.template pack<3>(coordTag, tidx);
        auto uv2 = (uv - bmin) / h;
        uv2[1] = 0;
        uv2[0] = zs::min(zs::max(uv2[0], 0.01f), nx - 1.01f);
        uv2[2] = zs::min(zs::max(uv2[2], 0.01f), ny - 1.01f);
        // uv2 = zeno::min(zeno::max(uv2, vec3f{0.01, 0.0, 0.01}), vec3f{nx - 1.01, 0.0, ny - 1.01});
        int i = uv2[0];
        int j = uv2[2];
        float cx = uv2[0] - i, cy = uv2[2] - j;
        size_t idx00 = j * nx + i, idx01 = j * nx + i + 1, idx10 = (j + 1) * nx + i, idx11 = (j + 1) * nx + i + 1;
        T f00 = prim.template pack<nchn>(fieldTag, idx00);
        T f01 = prim.template pack<nchn>(fieldTag, idx01);
        T f10 = prim.template pack<nchn>(fieldTag, idx10);
        T f11 = prim.template pack<nchn>(fieldTag, idx11);
        temp[tidx] = lerp<T>(lerp<T>(f00, f01, cx), lerp<T>(f10, f11, cx), cy);
    });

    pol(range(prim.size()), [temp = proxy<space>(temp), prim = proxy<space>({}, prim), fieldTag] ZS_LAMBDA(
                                int tidx) mutable { prim.template tuple<nchn>(fieldTag, tidx) = temp[tidx]; });
}
struct ZSGrid2DSample : zeno::INode {
    virtual void apply() override {
        using vec3f = zs::vec<float, 3>;
        auto nx = get_input2<int>("nx");
        auto ny = get_input2<int>("ny");
        auto bmin = get_input2<zeno::vec3f>("bmin");
        auto grid = get_input<ZenoParticles>("grid");
        auto attrT = get_input2<std::string>("attrT");
        auto channel = get_input2<std::string>("channel");
        auto sampleby = get_input2<std::string>("sampleBy");
        auto h = get_input2<float>("h");

        auto &pars = grid->getParticles();
        if (pars.hasProperty(channel) && pars.hasProperty(sampleby)) {
            if (attrT == "float") {
                sample2D<1>(pars, sampleby, channel, nx, ny, h, vec3f{bmin[0], bmin[1], bmin[2]});
            } else if (attrT == "vec3f") {
                sample2D<3>(pars, sampleby, channel, nx, ny, h, vec3f{bmin[0], bmin[1], bmin[2]});
            }
        }

        set_output("prim", std::move(grid));
    }
};
ZENDEFNODE(ZSGrid2DSample, {
                               {{"ZenoParticles", "grid"},
                                {"int", "nx", "1"},
                                {"int", "ny", "1"},
                                {"float", "h", "1"},
                                {"vec3f", "bmin", "0,0,0"},
                                {"string", "channel", "pos"},
                                {"string", "sampleBy", "pos"},
                                {"enum vec3 float", "attrT", "float"}},
                               {{"ZenoParticles", "prim"}},
                               {},
                               {"zenofx"},
                           });

} // namespace zeno