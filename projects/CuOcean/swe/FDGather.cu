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

} // namespace zeno