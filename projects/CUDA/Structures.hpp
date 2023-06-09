#pragma once
#include "zensim/container/Bht.hpp"
#include "zensim/container/Bvh.hpp"
#include "zensim/container/HashTable.hpp"
#include "zensim/container/IndexBuckets.hpp"
#include "zensim/container/SpatialHash.hpp"
#include "zensim/geometry/AnalyticLevelSet.h"
#include "zensim/geometry/Collider.h"
#include "zensim/geometry/SparseGrid.hpp"
#include "zensim/geometry/SparseLevelSet.hpp"
#include "zensim/geometry/Structure.hpp"
#include "zensim/geometry/Structurefree.hpp"
#include "zensim/physics/ConstitutiveModel.hpp"
#include "zensim/physics/constitutive_models/AnisotropicArap.hpp"
#include "zensim/physics/constitutive_models/EquationOfState.hpp"
#include "zensim/physics/constitutive_models/FixedCorotated.h"
#include "zensim/physics/constitutive_models/NeoHookean.hpp"
#include "zensim/physics/constitutive_models/StvkWithHencky.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeCamClay.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeDruckerPrager.hpp"
#include "zensim/physics/plasticity_models/NonAssociativeVonMises.hpp"
#include "zensim/resource/Resource.h"
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/zeno.h>

namespace zs {
template <typename T>
constexpr bool always_false = false;
}
namespace zeno {

using ElasticModel = zs::variant<zs::FixedCorotated<float>, zs::NeoHookean<float>, zs::StvkWithHencky<float>,
                                 zs::StableNeohookeanInvarient<float>>;
using AnisoElasticModel = zs::variant<std::monostate, zs::AnisotropicArap<float>>;
using PlasticModel = zs::variant<std::monostate, zs::NonAssociativeDruckerPrager<float>,
                                 zs::NonAssociativeVonMises<float>, zs::NonAssociativeCamClay<float>>;

struct ZenoConstitutiveModel : IObjectClone<ZenoConstitutiveModel> {
    enum elastic_model_e
    {
        Fcr,
        Nhk,
        Stvk
    };
    enum aniso_plastic_model_e
    {
        None_,
        Arap
    };
    enum plastic_model_e
    {
        None,
        DruckerPrager,
        VonMises,
        CamClay
    };

    enum config_value_type_e
    {
        Scalar,
        Vec3
    };
    static constexpr auto scalar_c = zs::wrapv<config_value_type_e::Scalar>{};
    static constexpr auto vec3_c = zs::wrapv<config_value_type_e::Vec3>{};
    using config_value_type = zs::variant<float, zs::vec<float, 3>>;

    auto &getElasticModel() noexcept {
        return elasticModel;
    }
    const auto &getElasticModel() const noexcept {
        return elasticModel;
    }
    auto &getAnisoElasticModel() noexcept {
        return anisoElasticModel;
    }
    const auto &getAnisoElasticModel() const noexcept {
        return anisoElasticModel;
    }
    auto &getPlasticModel() noexcept {
        return plasticModel;
    }
    const auto &getPlasticModel() const noexcept {
        return plasticModel;
    }

    template <elastic_model_e I>
    auto &getElasticModel() noexcept {
        return std::get<I>(elasticModel);
    }
    template <elastic_model_e I>
    const auto &getElasticModel() const noexcept {
        return std::get<I>(elasticModel);
    }
    template <plastic_model_e I>
    auto &getPlasticModel() noexcept {
        return std::get<I>(plasticModel);
    }
    template <plastic_model_e I>
    const auto &getPlasticModel() const noexcept {
        return std::get<I>(plasticModel);
    }

    template <typename T>
    void record(const std::string &tag, T &&val) {
        configs[tag] = val;
    }
    template <auto e = config_value_type_e::Scalar>
    auto retrieve(const std::string &tag, zs::wrapv<e> = {}) const {
        if constexpr (e == Scalar)
            return std::get<float>(configs.at(tag));
        else if constexpr (e == Vec3)
            return std::get<zs::vec<float, 3>>(configs.at(tag));
    }

    bool hasAnisoModel() const noexcept {
        return !std::holds_alternative<std::monostate>(anisoElasticModel);
    }
    bool hasPlasticModel() const noexcept {
        return !std::holds_alternative<std::monostate>(plasticModel);
    }

    bool hasF() const noexcept {
        return elasticModel.index() < 3;
    }
    bool hasLogJp() const noexcept {
        return plasticModel.index() == CamClay;
    }
    bool hasOrientation() const noexcept {
        return anisoElasticModel.index() != None;
    }
    bool hasPlasticity() const noexcept {
        return plasticModel.index() != None;
    }

    float dx, volume, density;
    ElasticModel elasticModel;
    AnisoElasticModel anisoElasticModel;
    PlasticModel plasticModel;
    std::map<std::string, config_value_type> configs;
};

struct ZenoParticles : IObjectClone<ZenoParticles> {
    // (i  ) traditional mpm particle,
    // (ii ) lagrangian mesh vertex particle
    // (iii) lagrangian mesh element quadrature particle
    // tracker particle for
    enum category_e : int
    {
        mpm,
        curve,
        surface,
        tet,
        tracker,
        vert_bending_spring,
        tri_bending_spring,
        bending
    };
    using particles_t = zs::TileVector<zs::f32, 32>;
    using dtiles_t = zs::TileVector<zs::f64, 32>;
    using vec3f = zs::vec<zs::f32, 3>;
    using vec3d = zs::vec<zs::f64, 3>;
    using bv_t = zs::AABBBox<3, zs::f32>;
    using lbvh_t = zs::LBvh<3, int, zs::f32>;

    static constexpr auto s_userDataTag = "userdata";
    static constexpr auto s_particleTag = "particles";
    static constexpr auto s_elementTag = "elements";
    static constexpr auto s_edgeTag = "edges";
    static constexpr auto s_surfTriTag = "surfaces";
    static constexpr auto s_surfEdgeTag = "surfEdges";
    static constexpr auto s_bendingEdgeTag = "bendingEdges";
    static constexpr auto s_surfVertTag = "surfVerts";
    static constexpr auto s_surfHalfEdgeTag = "surfHalfEdges";
    static constexpr auto s_tetHalfFacetTag = "tetHalfFacets";

    ZenoParticles() = default;
    ~ZenoParticles() = default;
    ZenoParticles(ZenoParticles &&o) noexcept = default;
    ZenoParticles(const ZenoParticles &o)
        : elements{o.elements}, category{o.category}, prim{o.prim}, model{o.model}, sprayedOffset{o.sprayedOffset},
          asBoundary{o.asBoundary} {
        if (o.particles)
            particles = std::make_shared<particles_t>(*o.particles);
    }
    ZenoParticles &operator=(ZenoParticles &&o) noexcept = default;
    ZenoParticles &operator=(const ZenoParticles &o) {
        ZenoParticles tmp{o};
        std::swap(*this, tmp);
        return *this;
    }

    template <bool HighPrecision = false>
    auto &getParticles(zs::wrapv<HighPrecision> = {}) {
        if constexpr (HighPrecision) {
            return images[s_particleTag];
        } else {
            if (!particles)
                throw std::runtime_error("particles (verts) not inited.");
            return *particles;
        }
    }
    template <bool HighPrecision = false>
    const auto &getParticles(zs::wrapv<HighPrecision> = {}) const {
        if constexpr (HighPrecision) {
            return images.at(s_particleTag);
        } else {
            if (!particles)
                throw std::runtime_error("particles (verts) not inited.");
            return *particles;
        }
    }
    template <bool HighPrecision = false>
    auto &getQuadraturePoints(zs::wrapv<HighPrecision> = {}) {
        if constexpr (HighPrecision) {
            return images[s_elementTag];
        } else {
            if (!elements.has_value())
                throw std::runtime_error("quadrature points not binded.");
            return *elements;
        }
    }
    template <bool HighPrecision = false>
    const auto &getQuadraturePoints(zs::wrapv<HighPrecision> = {}) const {
        if constexpr (HighPrecision) {
            return images.at(s_elementTag);
        } else {
            if (!elements.has_value())
                throw std::runtime_error("quadrature points not binded.");
            return *elements;
        }
    }
    bool isBendingString() const noexcept {
        return particles && elements.has_value() &&
               (category == category_e::vert_bending_spring || category == category_e::tri_bending_spring ||
                category == category_e::bending);
    }
    bool isMeshPrimitive() const noexcept {
        return particles && elements.has_value() &&
               (category == category_e::curve || category == category_e::surface || category == category_e::tet ||
                category == category_e::tracker);
    }
    bool isLagrangianParticles() const noexcept {
        return particles && elements.has_value() &&
               (category == category_e::curve || category == category_e::surface || category == category_e::tet);
    }
    auto &getModel() noexcept {
        return model;
    }
    const auto &getModel() const noexcept {
        return model;
    }

    int numDegree() const noexcept {
        switch (category) {
        case category_e::mpm: return 1;
        case category_e::curve: return 2;
        case category_e::surface: return 3;
        case category_e::tet: return 4;
        case category_e::tracker: return 0;
        default: return -1;
        }
        return -1;
    }
    std::size_t numParticles() const noexcept {
        return (*particles).size();
    }
    std::size_t numElements() const noexcept {
        return (*elements).size();
    }
    bool hasSprayedParticles() const noexcept {
        return sprayedOffset != numParticles();
    }

    decltype(auto) operator[](const std::string &tag) {
        return auxData[tag];
    }
    decltype(auto) operator[](const std::string &tag) const {
        return auxData.at(tag);
    }
    bool hasAuxData(const std::string &tag) const {
        // return auxData.find(tag) != auxData.end();
        if (auto it = auxData.find(tag); it != auxData.end())
            if (it->second.size() != 0)
                return true;
        return false;
    }
    decltype(auto) bvh(const std::string &tag) {
        return auxSpatialData[tag];
    }
    decltype(auto) bvh(const std::string &tag) const {
        return auxSpatialData.at(tag);
    }
    template <typename T>
    decltype(auto) setMeta(const std::string &tag, T &&val) {
        return metas[tag] = FWD(val);
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) const {
        return std::any_cast<T>(metas.at(tag));
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) {
        return std::any_cast<T>(metas.at(tag));
    }
    const UserData &zsUserData() const {
        return readMeta(s_userDataTag, zs::wrapt<const UserData &>{});
    }
    UserData &zsUserData() {
        return readMeta(s_userDataTag, zs::wrapt<UserData &>{});
    }
    bool hasBvh(const std::string &tag) const {
        // return auxSpatialData.find(tag) != auxSpatialData.end();
        if (auto it = auxSpatialData.find(tag); it != auxSpatialData.end())
            if (it->second.getNumLeaves() != 0)
                return true;
        return false;
    }
    bool hasMeta(const std::string &tag) const {
        if (auto it = metas.find(tag); it != metas.end())
            return true;
        return false;
    }
    bool hasImage(const std::string &tag) const {
        if (auto it = images.find(tag); it != images.end())
            if (it->second.size() != 0)
                return true;
        return false;
    }

    struct Mapping {
        zs::Vector<int> originalToOrdered, orderedToOriginal;
    };
    std::optional<Mapping> vertMapping, eleMapping;

    bool hasVertexMapping() const noexcept {
        return vertMapping.has_value();
    }
    Mapping &refVertexMapping() noexcept {
        return *vertMapping;
    }
    const Mapping &refVertexMapping() const noexcept {
        return *vertMapping;
    }

    template <typename Pol>
    bv_t computeBoundingVolume(Pol &pol, zs::SmallString xtag) const {
        using namespace zs;
        constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
        constexpr auto defaultBv =
            bv_t{vec3f::constant(zs::limits<zs::f32>::max()), vec3f::constant(zs::limits<zs::f32>::lowest())};
        if (!particles)
            return defaultBv;

        const auto &pars = *particles;
        zs::Vector<bv_t> bv{pars.get_allocator(), 1};
        bv.setVal(defaultBv);

        zs::Vector<zs::f32> X{pars.get_allocator(), pars.size()}, Y{pars.get_allocator(), pars.size()},
            Z{pars.get_allocator(), pars.size()};
        zs::Vector<zs::f32> res{pars.get_allocator(), 6};
        pol(enumerate(X, Y, Z), [pars = view<space>({}, pars), xOffset = pars.getPropertyOffset(xtag)] ZS_LAMBDA(
                                    int i, zs::f32 &x, zs::f32 &y, zs::f32 &z) mutable {
            auto xn = pars.pack(dim_c<3>, xOffset, i);
            x = xn[0];
            y = xn[1];
            z = xn[2];
        });
        zs::reduce(pol, std::begin(X), std::end(X), std::begin(res), zs::limits<zs::f32>::max(), getmin<zs::f32>{});
        zs::reduce(pol, std::begin(X), std::end(X), std::begin(res) + 3, zs::limits<zs::f32>::lowest(),
                   getmax<zs::f32>{});
        zs::reduce(pol, std::begin(Y), std::end(Y), std::begin(res) + 1, zs::limits<zs::f32>::max(), getmin<zs::f32>{});
        zs::reduce(pol, std::begin(Y), std::end(Y), std::begin(res) + 4, zs::limits<zs::f32>::lowest(),
                   getmax<zs::f32>{});
        zs::reduce(pol, std::begin(Z), std::end(Z), std::begin(res) + 2, zs::limits<zs::f32>::max(), getmin<zs::f32>{});
        zs::reduce(pol, std::begin(Z), std::end(Z), std::begin(res) + 5, zs::limits<zs::f32>::lowest(),
                   getmax<zs::f32>{});
        res = res.clone({memsrc_e::host, -1});
        return bv_t{vec3f{res[0], res[1], res[2]}, vec3f{res[3], res[4], res[5]}};
    }

    template <typename Pol>
    void updateElementIndices(Pol &pol, particles_t &eles) {
        using namespace zs;
        constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
        auto &dsts = (*vertMapping).originalToOrdered;
        auto nchns = eles.getPropertySize("inds");
        pol(range(eles.size()), [dsts = view<space>(dsts), eles = view<space>(eles),
                                 idOffset = eles.getPropertyOffset("inds"), nchns = nchns] ZS_LAMBDA(int i) mutable {
            for (int d = 0; d != nchns; ++d)
                eles(idOffset + d, i, int_c) = dsts[eles(idOffset + d, i, int_c)];
        });
    }
    template <typename Pol, typename CodeRange, typename IndexRange>
    void computeElementMortonCodes(Pol &pol, const bv_t &bv, const particles_t &eles, CodeRange &&codes,
                                   IndexRange &&indices) {
        using namespace zs;
        constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
        const auto &pars = *particles;
        auto nchns = eles.getPropertySize("inds");
        pol(range(eles.size()),
            [codes = std::begin(codes), indices = std::begin(indices), offset = eles.getPropertyOffset("inds"),
             eles = view<space>(eles), verts = view<space>({}, pars), bv = bv, nchns] ZS_LAMBDA(int ei) mutable {
                auto c = vec3f::zeros();
                for (int d = 0; d != nchns; ++d)
                    c += verts.pack(dim_c<3>, "x", eles(offset + d, ei, int_c));
                c /= nchns;
                auto coord = bv.getUniformCoord(c).template cast<f32>();
                codes[ei] = (zs::u32)morton_code<3>(coord);
                indices[ei] = ei;
            });
    }

    template <typename Pol, bool IncludeElement = true>
    void orderByMortonCode(Pol &pol, const bv_t &bv, zs::wrapv<IncludeElement> = {}) {
        using namespace zs;
        constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
        if (!particles)
            return;

        const auto &pars = *particles;
        if (!vertMapping.has_value()) {
            auto originalToOrdered = zs::Vector<int>{pars.get_allocator(), pars.size()};
            auto orderedToOriginal = zs::Vector<int>{pars.get_allocator(), pars.size()};
            vertMapping = Mapping{std::move(originalToOrdered), std::move(orderedToOriginal)};
        }
        auto &dsts = (*vertMapping).originalToOrdered;
        auto &indices = (*vertMapping).orderedToOriginal;
        dsts.resize(pars.size());
        indices.resize(pars.size());
        zs::Vector<zs::u32> tempBuffer{pars.get_allocator(), pars.size() * 2};
        pol(range(pars.size()), [dsts = view<space>(dsts), codes = view<space>(tempBuffer),
                                 verts = view<space>({}, pars), bv = bv] ZS_LAMBDA(int i) mutable {
            auto coord = bv.getUniformCoord(verts.pack(dim_c<3>, "x", i)).template cast<f32>();
            codes[i] = (zs::u32)morton_code<3>(coord);
            dsts[i] = i;
        });
        // radix sort
        radix_sort_pair(pol, std::begin(tempBuffer), dsts.begin(), std::begin(tempBuffer) + pars.size(),
                        indices.begin(), pars.size());
        // compute inverse mapping
        pol(range(pars.size()), [dsts = view<space>(dsts), indices = view<space>(indices)] ZS_LAMBDA(int i) mutable {
            dsts[indices[i]] = i;
        });
        // sort vert data
        auto verts = std::make_shared<particles_t>(pars.get_allocator(), pars.getPropertyTags(), pars.size());
        pol(range(pars.size()), [indices = view<space>(indices), verts = view<space>(*verts), pars = view<space>(pars),
                                 nchns = (int)pars.numChannels()] ZS_LAMBDA(int i) mutable {
            auto srcNo = indices[i];
            for (int d = 0; d != nchns; ++d)
                verts(d, i) = pars(d, srcNo);
        });
        particles = std::move(verts);
        if (hasImage(s_particleTag)) {
            auto &dpars = getParticles(true_c);
            auto dverts = dtiles_t{dpars.get_allocator(), dpars.getPropertyTags(), dpars.size()};
            pol(range(dpars.size()),
                [indices = view<space>(indices), dverts = view<space>(dverts), dpars = view<space>(dpars),
                 nchns = (int)dpars.numChannels()] ZS_LAMBDA(int i) mutable {
                    auto srcNo = indices[i];
                    for (int d = 0; d != nchns; ++d)
                        dverts(d, i) = dpars(d, srcNo);
                });
            images[s_particleTag] = std::move(dverts);
        }
        // update indices (modified in-place)
        bool flag = false;
        if (category == category_e::curve || category == category_e::surface || category == category_e::tet)
            flag = true;
        if (elements.has_value()) {
            auto &eles = getQuadraturePoints();
            if (flag) {
                updateElementIndices(pol, eles);
                if (hasImage(s_elementTag)) {
                    throw std::runtime_error("vertex mapping not enough for high-precision element index mapping\n");
                    // auto &deles = getQuadraturePoints(true_c);
                }
            }
        }
        if (flag) {
            if (hasAuxData(s_edgeTag))
                updateElementIndices(pol, operator[](s_edgeTag));
            else if (hasAuxData(s_surfTriTag))
                updateElementIndices(pol, operator[](s_surfTriTag));
            else if (hasAuxData(s_surfEdgeTag))
                updateElementIndices(pol, operator[](s_surfEdgeTag));
            else if (hasAuxData(s_surfVertTag))
                updateElementIndices(pol, operator[](s_surfVertTag));
            else if (hasAuxData(s_bendingEdgeTag))
                updateElementIndices(pol, operator[](s_bendingEdgeTag));
            else if (hasAuxData(s_surfHalfEdgeTag))
                updateElementIndices(pol, operator[](s_surfHalfEdgeTag));
        }

        if constexpr (!IncludeElement)
            return;
        {
            auto &eles = getQuadraturePoints();
            if (!eleMapping.has_value()) {
                auto originalToOrdered = zs::Vector<int>{eles.get_allocator(), eles.size()};
                auto orderedToOriginal = zs::Vector<int>{eles.get_allocator(), eles.size()};
                eleMapping = Mapping{std::move(originalToOrdered), std::move(orderedToOriginal)};
            }
            auto &dsts = (*eleMapping).originalToOrdered;
            auto &indices = (*eleMapping).orderedToOriginal;
            dsts.resize(eles.size());
            indices.resize(eles.size());
            tempBuffer.resize(eles.size() * 2);
            computeElementMortonCodes(pol, bv, eles, tempBuffer, dsts);
            // radix sort
            radix_sort_pair(pol, std::begin(tempBuffer), dsts.begin(), std::begin(tempBuffer) + eles.size(),
                            indices.begin(), eles.size());
            // compute inverse mapping
            pol(range(eles.size()), [dsts = view<space>(dsts), indices = view<space>(indices)] ZS_LAMBDA(
                                        int i) mutable { dsts[indices[i]] = i; });
            auto newEles = particles_t{eles.get_allocator(), eles.getPropertyTags(), eles.size()};
            pol(range(eles.size()),
                [indices = view<space>(indices), newEles = view<space>(newEles), eles = view<space>(eles),
                 nchns = (int)eles.numChannels()] ZS_LAMBDA(int i) mutable {
                    auto srcNo = indices[i];
                    for (int d = 0; d != nchns; ++d)
                        newEles(d, i) = eles(d, srcNo);
                });
            eles = std::move(newEles);
        }
    }

    std::shared_ptr<particles_t> particles{};
    std::optional<particles_t> elements{};
    std::map<std::string, particles_t> auxData;
    std::map<std::string, lbvh_t> auxSpatialData;
    std::map<std::string, std::any> metas;
    std::map<std::string, dtiles_t> images;
    category_e category{category_e::mpm}; // 0: conventional mpm particle, 1:
                                          // curve, 2: surface, 3: tet
    std::shared_ptr<PrimitiveObject> prim{};
    ZenoConstitutiveModel model{};
    std::size_t sprayedOffset{};
    bool asBoundary = false;
};

struct ZenoPartition : IObjectClone<ZenoPartition> {
    using Ti = int; // entry count
    using table_t = zs::bht<int, 3, int, 16>;
    using tag_t = zs::Vector<int>;
    using indices_t = zs::Vector<Ti>;

    auto &get() noexcept {
        return table;
    }
    const auto &get() const noexcept {
        return table;
    }

    auto numEntries() const noexcept {
        return table.size();
    }
    auto numBoundaryEntries() const noexcept {
        return (*boundaryIndices).size();
    }

    bool hasTags() const noexcept {
        return tags.has_value() && boundaryIndices.has_value();
    }
    auto &getTags() {
        return *tags;
    }
    const auto &getTags() const {
        return *tags;
    }
    auto &getBoundaryIndices() {
        return *boundaryIndices;
    }
    const auto &getBoundaryIndices() const {
        return *boundaryIndices;
    }

    void reserveTags() {
        auto numEntries = (std::size_t)table.size();
        if (!hasTags()) {
            tags = tag_t{numEntries, zs::memsrc_e::device, 0};
            boundaryIndices = indices_t{numEntries, zs::memsrc_e::device, 0};
        }
    }
    void clearTags() {
        if (hasTags())
            (*tags).reset(0);
    }

    table_t table;
    zs::optional<tag_t> tags;
    zs::optional<indices_t> boundaryIndices;
    bool requestRebuild{false};
    bool rebuilt;
};

struct ZenoGrid : IObjectClone<ZenoGrid> {
    enum transfer_scheme_e
    {
        Empty,
        Apic,
        Flip,
        AsFlip
    };
    using grid_t = zs::Grid<float, 3, 4, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
    auto &get() noexcept {
        return grid;
    }
    const auto &get() const noexcept {
        return grid;
    }

    bool isBoundaryGrid() const noexcept {
        return transferScheme == "boundary";
    }
    bool isPicStyle() const noexcept {
        return transferScheme == "apic";
    }
    bool hasPositionalAdjustment() const noexcept {
        return transferScheme == "sflip" || transferScheme == "asflip";
    }
    bool isFlipStyle() const noexcept {
        return transferScheme == "flip" || transferScheme == "aflip" || transferScheme == "sflip" ||
               transferScheme == "asflip";
    }
    bool isAffineAugmented() const noexcept {
        return transferScheme == "apic" || transferScheme == "aflip" || transferScheme == "asflip";
    }

    grid_t grid;
    std::string transferScheme; //
    std::shared_ptr<ZenoPartition> partition;
};

struct ZenoIndexBuckets : IObjectClone<ZenoIndexBuckets> {
    using buckets_t = zs::IndexBuckets<3, int, int, zs::grid_e::collocated, zs::ZSPmrAllocator<false>>;
    auto &get() noexcept {
        return ibs;
    }
    const auto &get() const noexcept {
        return ibs;
    }
    buckets_t ibs;
};

struct ZenoLinearBvh : IObjectClone<ZenoLinearBvh> {
    enum element_e
    {
        point,
        curve,
        surface,
        tet
    };
    using lbvh_t = zs::LBvh<3, int, zs::f32, zs::ZSPmrAllocator<false>>;
    auto &get() noexcept {
        return bvh;
    }
    const auto &get() const noexcept {
        return bvh;
    }
    lbvh_t bvh;
    zs::f32 thickness;
    element_e et{point};
};

struct ZenoSpatialHash : IObjectClone<ZenoSpatialHash> {
    enum element_e
    {
        point,
        curve,
        surface,
        tet
    };
    using sh_t = zs::SpatialHash<3, int, zs::f32, zs::ZSPmrAllocator<false>>;
    auto &get() noexcept {
        return sh;
    }
    const auto &get() const noexcept {
        return sh;
    }
    sh_t sh;
    zs::f32 thickness;
    element_e et{point};
};

struct ZenoLevelSet : IObjectClone<ZenoLevelSet> {
    // this supports a wide range of grid types (not just collocated)
    // default channel contains "sdf"
    // default transfer scheme is "unknown"
    using basic_ls_t = zs::BasicLevelSet<float, 3>;
    using const_sdf_vel_ls_t = zs::ConstSdfVelFieldPtr<float, 3>;
    using const_transition_ls_t = zs::ConstTransitionLevelSetPtr<float, 3>;
    using levelset_t = zs::variant<basic_ls_t, const_sdf_vel_ls_t, const_transition_ls_t>;

    template <zs::grid_e category = zs::grid_e::collocated>
    using spls_t = typename basic_ls_t::template spls_t<category>;
    using clspls_t = typename basic_ls_t::clspls_t;
    using ccspls_t = typename basic_ls_t::ccspls_t;
    using sgspls_t = typename basic_ls_t::sgspls_t;
    using dummy_ls_t = typename basic_ls_t::dummy_ls_t;
    using uniform_vel_ls_t = typename basic_ls_t::uniform_vel_ls_t;

    auto &getLevelSet() noexcept {
        return levelset;
    }
    const auto &getLevelSet() const noexcept {
        return levelset;
    }

    bool holdsBasicLevelSet() const noexcept {
        return std::holds_alternative<basic_ls_t>(levelset);
    }
    template <zs::grid_e category = zs::grid_e::collocated>
    bool holdsSparseLevelSet(zs::wrapv<category> = {}) const noexcept {
        return zs::match([](const auto &ls) {
            if constexpr (zs::is_same_v<RM_CVREF_T(ls), basic_ls_t>)
                return ls.template holdsLevelSet<spls_t<category>>();
            else
                return false;
        })(levelset);
    }
    decltype(auto) getBasicLevelSet() const noexcept {
        return std::get<basic_ls_t>(levelset);
    }
    decltype(auto) getBasicLevelSet() noexcept {
        return std::get<basic_ls_t>(levelset);
    }
    decltype(auto) getSdfVelField() const noexcept {
        return std::get<const_sdf_vel_ls_t>(levelset);
    }
    decltype(auto) getSdfVelField() noexcept {
        return std::get<const_sdf_vel_ls_t>(levelset);
    }
    decltype(auto) getLevelSetSequence() const noexcept {
        return std::get<const_transition_ls_t>(levelset);
    }
    decltype(auto) getLevelSetSequence() noexcept {
        return std::get<const_transition_ls_t>(levelset);
    }
    template <zs::grid_e category = zs::grid_e::collocated>
    decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) const noexcept {
        return std::get<basic_ls_t>(levelset).template getLevelSet<spls_t<category>>();
    }
    template <zs::grid_e category = zs::grid_e::collocated>
    decltype(auto) getSparseLevelSet(zs::wrapv<category> = {}) noexcept {
        return std::get<basic_ls_t>(levelset).template getLevelSet<spls_t<category>>();
    }

    levelset_t levelset;
    std::string transferScheme;
};

struct ZenoSparseGrid : IObjectClone<ZenoSparseGrid> {
    template <int level = 0>
    using grid_t = zs::SparseGrid<3, zs::f32, (8 >> level)>;
    using spg_t = grid_t<0>;

    auto &getSparseGrid() noexcept {
        return spg;
    }
    const auto &getSparseGrid() const noexcept {
        return spg;
    }

    template <typename T>
    decltype(auto) setMeta(const std::string &tag, T &&val) {
        return metas[tag] = FWD(val);
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) const {
        return std::any_cast<T>(metas.at(tag));
    }
    template <typename T = float>
    decltype(auto) readMeta(const std::string &tag, zs::wrapt<T> = {}) {
        return std::any_cast<T>(metas.at(tag));
    }
    bool hasMeta(const std::string &tag) const {
        if (auto it = metas.find(tag); it != metas.end())
            return true;
        return false;
    }
    /// @note -1 implies not a double buffer; 0/1 indicates the current double buffer index.
    int checkDoubleBuffer(const std::string &attr) const {
        auto metaTag = attr + "_cur";
        if (hasMeta(metaTag))
            return readMeta<int>(metaTag);
        return -1;
    }
    bool isDoubleBufferAttrib(const std::string &attr) const {
        if (attr.back() == '0' || attr.back() == '1')
            return true;
        else if (hasMeta(attr + "_cur"))
            return true;
        return false;
    }

    template <int level = 0>
    auto &getLevel() {
        if constexpr (level == 0)
            return spg;
        else if constexpr (level == 1)
            return spg1;
        else if constexpr (level == 2)
            return spg2;
        else if constexpr (level == 3)
            return spg3;
        else
            return spg;
    }

    spg_t spg;
    std::map<std::string, std::any> metas;

    grid_t<1> spg1;
    grid_t<2> spg2;
    grid_t<3> spg3;
};

struct ZenoBoundary : IObjectClone<ZenoBoundary> {
    using levelset_t = typename ZenoLevelSet::levelset_t;

    template <typename LsView>
    auto getBoundary(LsView &&lsv) const noexcept {
        using namespace zs;
        auto ret = Collider{FWD(lsv), type};
        ret.s = s;
        ret.dsdt = dsdt;
        ret.R = R;
        ret.omega = omega;
        ret.b = b;
        ret.dbdt = dbdt;
        return ret;
    }

    // levelset_t *levelset{nullptr};
    std::shared_ptr<ZenoLevelSet> zsls{};
    zs::collider_e type{zs::collider_e::Sticky};
    /** scale **/
    float s{1};
    float dsdt{0};
    /** rotation **/
    zs::Rotation<float, 3> R{zs::Rotation<float, 3>::identity()};
    zs::AngularVelocity<float, 3> omega{};
    /** translation **/
    zs::vec<float, 3> b{zs::vec<float, 3>::zeros()};
    zs::vec<float, 3> dbdt{zs::vec<float, 3>::zeros()};
};

template <typename Pol, int codim, typename MarkIter>
void mark_surface_boundary_verts(Pol &&pol, const ZenoParticles::particles_t &surf, zs::wrapv<codim>, MarkIter &&marks,
                                 int vOffset = 0) {
    using namespace zs;
    constexpr execspace_e space = RM_CVREF_T(pol)::exec_tag::value;
    auto allocator = get_temporary_memory_source(pol);
    using key_t = zs::vec<int, codim - 1>;
    bht<int, codim - 1, int, 16> tab{allocator, surf.size() * codim * 2};
    pol(range(surf.size()), [tab = view<space>(tab), surf = view<space>(surf),
                             indsOffset = surf.getPropertyOffset("inds")] ZS_LAMBDA(int ei) mutable {
        constexpr int dim = codim - 1;
        int i = surf(indsOffset, ei, int_c);
        for (int d = 0; d != dim; ++d) {
            int j = surf(indsOffset + (d + 1) % dim, ei, int_c);
            tab.insert(key_t{i, j});
            if (dim == 2)
                break;
            i = j;
        }
    });
    auto surfMarks = zs::Vector<int>{allocator, surf.size()};
    surfMarks.reset(0);
    pol(range(surf.size()), [tab = view<space>(tab), surf = view<space>(surf), surfMarks = view<space>(surfMarks),
                             indsOffset = surf.getPropertyOffset("inds")] ZS_LAMBDA(int ei) mutable {
        using tab_t = RM_CVREF_T(tab);
        constexpr int dim = codim - 1;
        int i = surf(indsOffset, ei, int_c);
        bool isBoundary = true;
        for (int d = 0; d != dim; ++d) {
            int j = surf(indsOffset + (d + 1) % dim, ei, int_c);
            if (tab.query(key_t{j, i}) != tab_t::sentinel_v) { // this surf element is not boundary
                isBoundary = false;
                break;
            }
            i = j;
        }
        surfMarks[ei] = isBoundary;
    });
    pol(range(surf.size()), [surf = view<space>(surf), surfMarks = view<space>(surfMarks), marks,
                             indsOffset = surf.getPropertyOffset("inds"), vOffset] ZS_LAMBDA(int ei) mutable {
        using tab_t = RM_CVREF_T(tab);
        constexpr int dim = codim - 1;
        if (!surfMarks[ei])
            return;
        for (int d = 0; d != codim; ++d) {
            int i = surf(indsOffset + d, ei, int_c);
            marks[i + vOffset] = 1;
        }
    });
}

} // namespace zeno