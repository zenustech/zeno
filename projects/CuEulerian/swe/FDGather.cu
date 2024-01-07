#include "Structures.hpp"
#include "zensim/Logger.hpp"
#include "zensim/cuda/execution/ExecutionPolicy.cuh"
#include "zensim/omp/execution/ExecutionPolicy.hpp"
#include "zensim/types/Property.h"
#include <zeno/types/ListObject.h>
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
        pol, {{lTag, nchn}, {rTag, nchn}, {tTag, nchn}, {bTag, nchn}});
    pol(Collapse{(std::size_t)ny * (std::size_t)nx},
        [verts = proxy<space>({}, prim), nx, ny, lTag, rTag, tTag, bTag, tag] ZS_LAMBDA(size_t id) mutable {
            int i = id % nx;
            int j = id / nx;
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
void checkEdgeLoop(PrimitiveObject *prim, typename ZenoParticles::particles_t &zsprim, int nx, int ny,
                   const std::string &channel) {
    using namespace zs;
    constexpr auto space = execspace_e::host;
    auto pol = omp_exec();
    using T = conditional_t<nchn == 1, float, zeno::vec3f>;
    auto &l = prim->add_attr<T>("l" + channel);
    auto &r = prim->add_attr<T>("r" + channel);
    auto &t = prim->add_attr<T>("t" + channel);
    auto &b = prim->add_attr<T>("b" + channel);
    auto &channels = prim->add_attr<T>(channel);

    const SmallString lTag = std::string("l") + channel;
    const SmallString rTag = std::string("r") + channel;
    const SmallString tTag = std::string("t") + channel;
    const SmallString bTag = std::string("b") + channel;
    const SmallString tag = channel;
    pol(Collapse{nx, ny}, [&, verts = proxy<space>({}, zsprim)](int i, int j) {
        size_t lidx = j * nx + math::max(i - 1, 0);
        size_t ridx = j * nx + math::min(i + 1, nx - 1);
        size_t tidx = math::min(j + 1, ny - 1) * nx + i;
        size_t bidx = math::max(j - 1, 0) * nx + i;
        size_t idx = j * nx + i;

        if constexpr (nchn == 1) {
            auto dev_results = [&]() {
                auto li = verts(lTag, idx);
                auto ri = verts(rTag, idx);
                auto ti = verts(tTag, idx);
                auto bi = verts(bTag, idx);
                auto inli = verts(tag, lidx);
                auto inri = verts(tag, ridx);
                auto inti = verts(tag, tidx);
                auto inbi = verts(tag, bidx);
                return zs::make_tuple(li, ri, ti, bi, inli, inri, inti, inbi);
            };
            auto host_results = [&]() {
                auto li = l[idx];
                auto ri = r[idx];
                auto ti = t[idx];
                auto bi = b[idx];
                auto inli = channels[lidx];
                auto inri = channels[ridx];
                auto inti = channels[tidx];
                auto inbi = channels[bidx];
                return zs::make_tuple(li, ri, ti, bi, inli, inri, inti, inbi);
            };
            auto [da, db, dc, dd, dain, dbin, dcin, ddin] = dev_results();
            auto [ha, hb, hc, hd, hain, hbin, hcin, hdin] = host_results();
            if (da != ha || db != hb || dc != hc || dd != hd || dain != hain || dbin != hbin || dcin != hcin ||
                ddin != hdin)
                printf("damn wrong at <%d, %d>!\n\tdev: [%f(%f), %f(%f), %f(%f), %f(%f)]\n\thost_ref: [%f(%f), %f(%f), "
                       "%f(%f), %f(%f)]\n",
                       i, j, da, dain, db, dbin, dc, dcin, dd, ddin, ha, hain, hb, hbin, hc, hcin, hd, hdin);
        } else if constexpr (nchn == 3) {
            auto dev_results = [&]() {
                auto li = verts.template pack<nchn>(lTag, idx).to_array();
                auto ri = verts.template pack<nchn>(rTag, idx).to_array();
                auto ti = verts.template pack<nchn>(tTag, idx).to_array();
                auto bi = verts.template pack<nchn>(bTag, idx).to_array();
                auto inli = verts.template pack<nchn>(tag, lidx).to_array();
                auto inri = verts.template pack<nchn>(tag, ridx).to_array();
                auto inti = verts.template pack<nchn>(tag, tidx).to_array();
                auto inbi = verts.template pack<nchn>(tag, bidx).to_array();
                return zs::make_tuple(li, ri, ti, bi, inli, inri, inti, inbi);
            };
            auto host_results = [&]() {
                auto li = l[idx];
                auto ri = r[idx];
                auto ti = t[idx];
                auto bi = b[idx];
                auto inli = channels[lidx];
                auto inri = channels[ridx];
                auto inti = channels[tidx];
                auto inbi = channels[bidx];
                return zs::make_tuple(li, ri, ti, bi, inli, inri, inti, inbi);
            };
            auto [da, db, dc, dd, dain, dbin, dcin, ddin] = dev_results();
            auto [ha, hb, hc, hd, hain, hbin, hcin, hdin] = host_results();
            if (da != ha || db != hb || dc != hc || dd != hd || dain != hain || dbin != hbin || dcin != hcin ||
                ddin != hdin)
                printf("damn wrong vec3f! <%d, %d>\n", i, j);
        }
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
    pol(Collapse{(size_t)ny * (size_t)nx},
        [verts = proxy<space>({}, prim), nx, ny, lTag, rTag, tTag, bTag, tag] ZS_LAMBDA(size_t id) mutable {
            int i = id % nx;
            int j = id / nx;
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
        pol, {{ltTag, nchn}, {rtTag, nchn}, {lbTag, nchn}, {rbTag, nchn}});
    pol(Collapse{(size_t)ny * (size_t)nx},
        [verts = proxy<space>({}, prim), nx, ny, ltTag, rtTag, lbTag, rbTag, tag] ZS_LAMBDA(size_t id) mutable {
            int i = id % nx;
            int j = id / nx;
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
    pol(Collapse{(size_t)ny * (size_t)nx},
        [verts = proxy<space>({}, prim), nx, ny, ltTag, rtTag, lbTag, rbTag, tag] ZS_LAMBDA(size_t id) mutable {
            int i = id % nx;
            int j = id / nx;
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

struct ZSCheckGather2DFiniteDifference : zeno::INode {
    virtual void apply() override {
        auto nx = get_input2<int>("nx");
        auto ny = get_input2<int>("ny");
        auto grid = get_input<ZenoParticles>("ZSParticles");
        auto gridRef = get_input<PrimitiveObject>("grid");
        auto attrT = get_input2<std::string>("attrT");
        auto type = get_input2<std::string>("OpType");
        auto channel = get_input2<std::string>("channel");

        if (auto &verts_ = grid->getParticles(); verts_.hasProperty(channel)) {
            auto verts = verts_.clone({zs::memsrc_e::host, -1});
            if (type == "FIVE_STENCIL" || type == "NINE_STENCIL") {
                puts("begin edge loop checking");
                if (attrT == "float") {
                    checkEdgeLoop<1>(gridRef.get(), verts, nx, ny, channel);
                }
                if (attrT == "vec3") {
                    checkEdgeLoop<3>(gridRef.get(), verts, nx, ny, channel);
                }
                puts("done edge loop checking");
            }
#if 0
            if (type == "NINE_STENCIL") {
                if (attrT == "float") {
                    checkCornerLoop<1>(gridRef.get(), verts, nx, ny, channel);
                }
                if (attrT == "vec3") {
                    checkCornerLoop<3>(gridRef.get(), verts, nx, ny, channel);
                }
            }
#endif
        }

        set_output("prim", std::move(grid));
    }
};

ZENDEFNODE(ZSCheckGather2DFiniteDifference, {
                                                {{"PrimitiveObject", "grid"},
                                                 {"ZSParticles", "ZSParticles"},
                                                 {"int", "nx", "1"},
                                                 {"int", "ny", "1"},
                                                 {"string", "channel", "pos"},
                                                 {"enum vec3 float", "attrT", "float"},
                                                 {"enum FIVE_STENCIL NINE_STENCIL", "OpType", "FIVE_STENCIL"}},
                                                {{"ZSParticles", "prim"}},
                                                {},
                                                {"zenofx"},
                                            });

struct ZSCheckPrimAttribs : zeno::INode {
    virtual void apply() override {
        auto grid = get_input<ZenoParticles>("ZSParticles");
        auto gridRef = get_input<PrimitiveObject>("grid");
        auto attribs = get_input<zeno::ListObject>("attribs");

        const auto &verts = grid->getParticles();
        for (auto &attrib_ : attribs->get2<std::string>()) {
            auto attrib = attrib_;
            auto attribAct = attrib_;
            if (attribAct == "pos")
                attribAct = "x";
            if (attribAct == "vel")
                attribAct = "v";
            if (!verts.hasProperty(attribAct) || !gridRef->has_attr(attrib))
                throw std::runtime_error(fmt::format("about prop [{}], zspar [{}], ref [{}]\n", attrib,
                                                     verts.hasProperty(attribAct), gridRef->has_attr(attrib)));
            if (verts.size() != gridRef->size())
                throw std::runtime_error("size mismatch!\n");
            auto nchn = verts.getPropertySize(attribAct);
            auto nchnRef = gridRef->attr_is<float>(attrib) ? 1 : 3;
            if (nchn != nchnRef)
                throw std::runtime_error("attrib dimension mismatch!\n");
            auto compare = [&](auto dim_v) {
                fmt::print("begin checking prim [{}] of size {}\n", attrib, nchn);
                using namespace zs;
                constexpr auto dim = RM_CVREF_T(dim_v)::value;
                using T = conditional_t<dim == 1, float, zeno::vec3f>;
                auto ompPol = omp_exec();
                constexpr auto space = execspace_e::host;
                ompPol(range(verts.size()),
                       [&, verts = proxy<space>({}, verts), tag = SmallString{attribAct}](int vi) {
                           float v, vref;
                           for (int d = 0; d != nchn; ++d) {
                               v = verts(tag, d, vi);
                               if constexpr (RM_CVREF_T(dim_v)::value == 1)
                                   vref = gridRef->attr<T>(attrib)[vi];
                               else
                                   vref = gridRef->attr<T>(attrib)[vi][d];
                               if (v != vref) {
                                   fmt::print("damn this, {}-th vert prop [{}, {}] actual: {}, ref: {}\n", vi, attrib,
                                              d, v, vref);
                               }
                           }
                       });
                fmt::print("done checking prim [{}]\n", attrib);
            };
            std::variant<zs::wrapv<1>, zs::wrapv<3>> tmp{};
            if (nchn == 1)
                tmp = zs::wrapv<1>{};
            else
                tmp = zs::wrapv<3>{};
            zs::match(compare)(tmp);
        }

        set_output("prim", std::move(grid));
    }
};

ZENDEFNODE(ZSCheckPrimAttribs,
           {
               {{"PrimitiveObject", "grid"}, {"ZSParticles", "ZSParticles"}, {"ListObject", "attribs"}},
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