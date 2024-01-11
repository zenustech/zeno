#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/zeno.h>
//#include <opensubdiv/far/topologyDescriptor.h>
//#include <opensubdiv/far/stencilTableFactory.h>
//#include <opensubdiv/osd/cpuEvaluator.h>
//#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <cstdio>
#include <cstring>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>

namespace zeno {
namespace {

using namespace OpenSubdiv;

//struct OSDParams {
//float *verts; // 3 * nverts
//int *vertsperface; // nfaces
//int *vertIndices; // vertsperface[0] + ... + vertsperface[nfaces - 1]
//int nverts;
//int nfaces;
//};
namespace {
struct Vertex3 {

    // Minimal required interface ----------------------
    Vertex3() {
    }

    void Clear(void * = 0) {
        _point[0] = _point[1] = _point[2] = 0.0f;
    }

    void AddWithWeight(Vertex3 const &src, float weight) {
        _point[0] += weight * src._point[0];
        _point[1] += weight * src._point[1];
        _point[2] += weight * src._point[2];
    }

    // Public interface ------------------------------------
    void SetPoint(float x, float y, float z) {
        _point[0] = x;
        _point[1] = y;
        _point[2] = z;
    }

    const float *GetPoint() const {
        return _point;
    }

  private:
    float _point[3];
};

struct Vertex2 {

    // Minimal required interface ----------------------
    Vertex2() {
    }

    void Clear(void * = 0) {
        _point[0] = _point[1] = 0.0f;
    }

    void AddWithWeight(Vertex2 const &src, float weight) {
        _point[0] += weight * src._point[0];
        _point[1] += weight * src._point[1];
    }

    // Public interface ------------------------------------
    void SetPoint(float x, float y) {
        _point[0] = x;
        _point[1] = y;
    }

    const float *GetPoint() const {
        return _point;
    }

  private:
    float _point[2];
};

struct Vertex1 {

    // Minimal required interface ----------------------
    Vertex1() {
    }

    void Clear(void * = 0) {
        _point[0] = 0.0f;
    }

    void AddWithWeight(Vertex1 const &src, float weight) {
        _point[0] += weight * src._point[0];
    }

    // Public interface ------------------------------------
    void SetPoint(float x, float y, float z) {
        _point[0] = x;
    }

    const float *GetPoint() const {
        return _point;
    }

  private:
    float _point[1];
};

static Vertex3 *convvertexptr(vec3f *p) {
    return reinterpret_cast<Vertex3 *>(p);
}

static Vertex2 *convvertexptr(vec2f *p) {
    return reinterpret_cast<Vertex2 *>(p);
}

static Vertex1 *convvertexptr(float *p) {
    return reinterpret_cast<Vertex1 *>(p);
}

static vec3f v2to3(vec2f const &v) {
    return {v[0], v[1], 0};
}
} // namespace

//------------------------------------------------------------------------------
static void osdPrimSubdiv(PrimitiveObject *prim, int levels, std::string edgeCreaseAttr = {}, bool triangulate = false,
                          bool asQuadFaces = false, bool hasLoopUVs = true, bool copyFaceAttrs = true) {
    const int maxlevel = levels;
    if (maxlevel <= 0 || !prim->verts.size())
        return;

    if (!(prim->loops.size() && prim->loops.has_attr("uvs")))
        hasLoopUVs = false;

    std::vector<int> polysInd, polysLen;
    int primpolyreduced = 0;
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len <= 2)
            continue;
        primpolyreduced += len;
    }

    int offsetred = 0;
    polysLen.resize(prim->polys.size());
    polysInd.resize(offsetred + primpolyreduced);
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len <= 2)
            continue;
        polysLen[i] = len;
        for (int j = 0; j < len; j++) {
            polysInd[offsetred + j] = prim->loops[base + j];
        }
        offsetred += len;
    }

    if (!polysLen.size() || !polysInd.size())
        return;

    Far::TopologyDescriptor desc;
    desc.numVertices = prim->verts.size();
    desc.numFaces = polysLen.size();
    desc.numVertsPerFace = polysLen.data();
    desc.vertIndicesPerFace = polysInd.data();
    if (edgeCreaseAttr.size()) {
        auto const &crease = prim->lines.attr<float>(edgeCreaseAttr);
        desc.numCreases = crease.size();
        desc.creaseVertexIndexPairs = reinterpret_cast<int const *>(prim->lines.data());
        desc.creaseWeights = crease.data();
    }

    std::vector<Far::TopologyDescriptor::FVarChannel> channels;
    std::vector<int> uvsInd;
    if (hasLoopUVs) {
        uvsInd.resize(polysInd.size());
        int offsetred = 0;
        auto &loop_uvs = prim->loops.attr<int>("uvs");
        for (int i = 0; i < prim->polys.size(); i++) {
            auto [base, len] = prim->polys[i];
            if (len <= 2)
                continue;
            for (int j = 0; j < len; j++) {
                uvsInd[offsetred + j] = loop_uvs[base + j];
            }
            offsetred += len;
        }

        auto &ch = channels.emplace_back();
        ch.numValues = uvsInd.size();
        ch.valueIndices = uvsInd.data();

        desc.numFVarChannels = channels.size();
        desc.fvarChannels = channels.data();
    }

    std::map<std::string, AttrVector<vec2i>::AttrVectorVariant> oldpolyattrs;
    if (copyFaceAttrs) { // make zhxx very happy
        size_t offsetred = 0, curOffsetred;
        size_t shift = 2 * (levels - 1);
        size_t finred = 0;
        for (size_t i = 0; i < prim->polys.size(); i++) {
            size_t stride = prim->polys[i][1] << shift;
            finred += stride;
        }
        auto fits = [&] (auto &pat) {
            if (pat.size() < finred) pat.resize(finred);
        };

        curOffsetred = offsetred;
        prim->polys.foreach_attr<AttrAcceptAll>([&](std::string const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &pat = oldpolyattrs[key].emplace<std::vector<T>>();
            fits(pat);

            auto offsetred = curOffsetred;
            for (size_t i = 0; i < prim->polys.size(); i++) {
                size_t stride = prim->polys[i][1] << shift;
                for (size_t j = 0; j < stride; j++) {
                    pat[offsetred + j] = arr[i];
                }
                offsetred += stride;
            }
        });
        offsetred += primpolyreduced;
    }

    prim->points.clear();
    prim->lines.clear();
    prim->tris.clear();
    prim->quads.clear();
    prim->polys.clear();
    prim->loops.clear();

    Sdc::SchemeType refinetfactype = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options refineofactptions;
    refineofactptions.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
    // Instantiate a Far::TopologyRefiner from the descriptor
    using Factory = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
    std::unique_ptr<Far::TopologyRefiner> refiner(
        Factory::Create(desc, Factory::Options(refinetfactype, refineofactptions)));
    if (!refiner)
        throw makeError("refiner is null (factory creation failed)");

    // Uniformly refine the topology up to 'maxlevel'
    // note: fullTopologyInLastLevel must be true to work with face-varying data
    {
        Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
        refineOptions.fullTopologyInLastLevel = hasLoopUVs;
        refiner->RefineUniform(refineOptions);
    }

    //// Allocate a buffer for vertex primvar data. The buffer length is set to
    //// be the sum of all children vertices up to the highest level of refinement.

    int nCoarseVerts = prim->verts.size();
    int nFineVerts = refiner->GetLevel(maxlevel).GetNumVertices();
    int nTotalVerts = refiner->GetNumVerticesTotal();
    int nTempVerts = nTotalVerts - nCoarseVerts - nFineVerts;
    prim->verts.resize(nCoarseVerts + nTempVerts);

    AttrVector<vec2f> fine_uvs;
    int nCoarseFVars{}, nFineFVars{}, nTotalFVars{}, nTempFVars{};
    if (hasLoopUVs) {
        nCoarseFVars = prim->uvs.size(); //channels[0].numValues;
        nFineFVars = refiner->GetLevel(maxlevel).GetNumFVarValues();
        nTotalFVars = refiner->GetNumFVarValuesTotal();
        nTempFVars = nTotalFVars - nCoarseFVars - nFineFVars;
        prim->uvs.resize(nCoarseFVars + nTempFVars);
        fine_uvs.resize(nFineFVars);
    }
    AttrVector<vec3f> fine_verts(nFineVerts);

    // Interpolate vertex primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    size_t srcposoffs = 0;
    size_t dstposoffs = nCoarseVerts;

    size_t srcfvaroffs{};
    size_t dstfvaroffs{};
    if (hasLoopUVs) {
        dstfvaroffs = nCoarseFVars;
    }

    for (int level = 1; level < maxlevel; ++level) {
        auto *srcPos = convvertexptr(prim->verts.data() + srcposoffs);
        auto *dstPos = convvertexptr(prim->verts.data() + dstposoffs);
        primvarRefiner.Interpolate(level, srcPos, dstPos);
        prim->verts.foreach_attr([&](auto const &key, auto &arr) {
            auto *srcClr = convvertexptr(arr.data() + srcposoffs);
            auto *dstClr = convvertexptr(arr.data() + dstposoffs);
            primvarRefiner.InterpolateVarying(level, srcClr, dstClr);
        });
        if (hasLoopUVs) {
            auto *srcFVarColor = convvertexptr(prim->uvs.data() + srcfvaroffs);
            auto *dstFVarColor = convvertexptr(prim->uvs.data() + dstfvaroffs);
            primvarRefiner.InterpolateFaceVarying(level, srcFVarColor, dstFVarColor);
            auto numfvars = refiner->GetLevel(level).GetNumFVarValues();
            srcfvaroffs = dstfvaroffs;
            dstfvaroffs += numfvars;
        }
        auto numverts = refiner->GetLevel(level).GetNumVertices();
        srcposoffs = dstposoffs;
        dstposoffs += numverts;
    }

    // Interpolate the last level into the separate buffers for our final data:
    {
        auto *srcPos = convvertexptr(prim->verts.data() + srcposoffs);
        auto *dstPos = convvertexptr(fine_verts.data());
        primvarRefiner.Interpolate(maxlevel, srcPos, dstPos);
        prim->verts.foreach_attr([&](auto const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &fine_arr = fine_verts.add_attr<T>(key);
            auto *srcClr = convvertexptr(arr.data() + srcposoffs);
            auto *dstClr = convvertexptr(fine_arr.data());
            primvarRefiner.InterpolateVarying(maxlevel, srcClr, dstClr);
        });
        if (hasLoopUVs) {
            auto *srcFVarColor = convvertexptr(prim->uvs.data() + srcfvaroffs);
            auto *dstFVarColor = convvertexptr(fine_uvs.data());
            primvarRefiner.InterpolateFaceVarying(maxlevel, srcFVarColor, dstFVarColor);
        }
    }

    { // Output OBJ of the highest level refined -----------

        Far::TopologyLevel const &refLastLevel = refiner->GetLevel(maxlevel);

        int nverts = refLastLevel.GetNumVertices();
        int nfaces = refLastLevel.GetNumFaces();

        int nfvars{};
        if (hasLoopUVs) {
            nfvars = refLastLevel.GetNumFVarValues();
        }

        std::swap(prim->verts, fine_verts);
        fine_verts.clear();
        fine_verts.shrink_to_fit();
        assert(prim->verts.size() == nverts);

        std::swap(prim->uvs, fine_uvs);
        fine_uvs.clear();
        fine_uvs.shrink_to_fit();
        assert(prim->uvs.size() == nfvars);

        // Print faces
        {
            prim->polys.resize(nfaces);
            prim->loops.resize(nfaces * 4);

            if (copyFaceAttrs) {
                for (auto const &[key_, atta] : oldpolyattrs) {
                    std::visit(
                        [&, key = key_](auto &arr) {
                            using T = std::decay_t<decltype(arr[0])>;
                            if (arr.size() != nfaces) {
                                zeno::log_warn("copyFaceAttrs estimated face count mismatch");
                            }
                            prim->polys.add_attr<T>(key) = std::move(arr);
                        },
                        atta);
                }
            }

            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size() == 4);

                prim->loops[face * 4 + 0] = fverts[0];
                prim->loops[face * 4 + 1] = fverts[1];
                prim->loops[face * 4 + 2] = fverts[2];
                prim->loops[face * 4 + 3] = fverts[3];
                prim->polys[face] = {face * 4, 4};
            }

            if (hasLoopUVs) {
                auto &loop_uvs = prim->loops.attr<int>("uvs");
                loop_uvs.resize(nfaces * 4);

                for (int face = 0; face < nfaces; ++face) {
                    Far::ConstIndexArray fvars = refLastLevel.GetFaceFVarValues(face);
                    assert(fvars.size() == 4);
                    loop_uvs[face * 4 + 0] = fvars[0];
                    loop_uvs[face * 4 + 1] = fvars[1];
                    loop_uvs[face * 4 + 2] = fvars[2];
                    loop_uvs[face * 4 + 3] = fvars[3];
                }
            }
        }
    }
}

struct OSDPrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int levels = get_input2<int>("levels");
        if (get_input2<bool>("delayTillIpc") && levels) { // cihou zhxx
            prim->userData().set2("delayedSubdivLevels", levels);
            set_output("prim", std::move(prim));
            return;
        }
        auto edgeCreaseAttr = get_input2<std::string>("edgeCreaseAttr");
        bool triangulate = get_input2<bool>("triangulate");
        bool asQuadFaces = get_input2<bool>("asQuadFaces");
        bool hasLoopUVs = get_input2<bool>("hasLoopUVs");
        bool copyFaceAttrs = get_input2<bool>("copyFaceAttrs");
        if (levels)
            osdPrimSubdiv(prim.get(), levels, edgeCreaseAttr, triangulate, asQuadFaces, hasLoopUVs, copyFaceAttrs);
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDPrimSubdiv)
({
    {
        "prim",
        {"int", "levels", "2"},
        {"string", "edgeCreaseAttr", ""},
        {"bool", "triangulate", "1"},
        {"bool", "asQuadFaces", "1"},
        {"bool", "hasLoopUVs", "1"},
        {"bool", "copyFaceAttrs", "1"},
        {"bool", "delayTillIpc", "0"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

//------------------------------------------------------------------------------

} // namespace
} // namespace zeno
