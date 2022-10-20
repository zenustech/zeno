#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/types/PrimitiveObject.h>
//#include <opensubdiv/far/topologyDescriptor.h>
//#include <opensubdiv/far/stencilTableFactory.h>
//#include <opensubdiv/osd/cpuEvaluator.h>
//#include <opensubdiv/osd/cpuVertexBuffer.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <cstring>
#include <cstdio>

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
        Vertex3() { }

        void Clear( void * =0 ) {
            _point[0]=_point[1]=_point[2]=0.0f;
        }

        void AddWithWeight(Vertex3 const & src, float weight) {
            _point[0]+=weight*src._point[0];
            _point[1]+=weight*src._point[1];
            _point[2]+=weight*src._point[2];
        }

        // Public interface ------------------------------------
        void SetPoint(float x, float y, float z) {
            _point[0]=x;
            _point[1]=y;
            _point[2]=z;
        }

        const float * GetPoint() const {
            return _point;
        }

    private:
        float _point[3];
    };

    struct Vertex2 {

        // Minimal required interface ----------------------
        Vertex2() { }

        void Clear( void * =0 ) {
            _point[0]=_point[1]=0.0f;
        }

        void AddWithWeight(Vertex2 const & src, float weight) {
            _point[0]+=weight*src._point[0];
            _point[1]+=weight*src._point[1];
        }

        // Public interface ------------------------------------
        void SetPoint(float x, float y) {
            _point[0]=x;
            _point[1]=y;
        }

        const float * GetPoint() const {
            return _point;
        }

    private:
        float _point[2];
    };

    struct Vertex1 {

        // Minimal required interface ----------------------
        Vertex1() { }

        void Clear( void * =0 ) {
            _point[0]=0.0f;
        }

        void AddWithWeight(Vertex1 const & src, float weight) {
            _point[0]+=weight*src._point[0];
        }

        // Public interface ------------------------------------
        void SetPoint(float x, float y, float z) {
            _point[0]=x;
        }

        const float * GetPoint() const {
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
}


//------------------------------------------------------------------------------
static void osdPrimSubdiv(PrimitiveObject *prim, int levels, std::string edgeCreaseAttr = {}, bool triangulate = false, bool asQuadFaces = false, bool hasLoopUVs = true) {
    const int maxlevel=levels;
    if (maxlevel <= 0 || !prim->verts.size()) return;

    if (prim->loops.size() && prim->loops.has_attr("uvs"))
        hasLoopUVs = false;

        //nCoarseVerts=0,
        //nRefinedVerts=0;
    //std::vector<int> ncfaces(maxlevel);
    //std::vector<int> ncedges(maxlevel);

    std::vector<int> polysInd, polysLen;
    int primpolyreduced = 0;
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len <= 2) continue;
        primpolyreduced += len;
    }
    polysLen.reserve(prim->tris.size() + prim->quads.size() + prim->polys.size());
    polysInd.reserve(prim->tris.size() * 3 + prim->quads.size() * 4 + primpolyreduced);

    polysLen.resize(prim->tris.size(), 3);
    polysInd.insert(polysInd.end(),
                     reinterpret_cast<int const *>(prim->tris.data()),
                     reinterpret_cast<int const *>(prim->tris.data() + prim->tris.size()));

    polysLen.resize(prim->tris.size() + prim->quads.size(), 4);
    polysInd.insert(polysInd.end(),
                     reinterpret_cast<int const *>(prim->quads.data()),
                     reinterpret_cast<int const *>(prim->quads.data() + prim->quads.size()));

    int offsetred = prim->tris.size() * 3 + prim->quads.size() * 4;
    polysLen.resize(prim->tris.size() + prim->quads.size() + prim->polys.size());
    polysInd.resize(offsetred + primpolyreduced);
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        if (len <= 2) continue;
        polysLen[prim->tris.size() + prim->quads.size() + i] = len;
        for (int j = 0; j < len; j++) {
            polysInd[offsetred + j] = prim->loops[base + j];
        }
        offsetred += len;
    }

    if (!polysLen.size() || !polysInd.size()) return;


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
    /*std::vector<std::string> chanveckeys;*/
    if (hasLoopUVs) {

        /*channels.reserve(prim->loops.num_attrs());*/
        /*loopsIndTab.reserve(prim->loops.num_attrs());*/
        /*chanveckeys.reserve(prim->loops.num_attrs());*/
        /*for (auto const &key: prim->loops.attr_keys()) {*/
            /*auto &loopsInd = loopsIndTab.emplace_back();*/
            uvsInd.resize(polysInd.size());
            int offsetred = prim->tris.size() * 3 + prim->quads.size() * 4;
            auto &loop_uvs = prim->loops.attr<int>("uvs");
            for (int i = 0; i < prim->polys.size(); i++) {
                auto [base, len] = prim->polys[i];
                if (len <= 2) continue;
                for (int j = 0; j < len; j++) {
                    uvsInd[offsetred + j] = loop_uvs[base + j];
                    //prim->loops.attr<int>(key)[base + j];
                }
                offsetred += len;
            }
            //if (key.size() >= 4 && key[0] == 'I' && key[1] == 'N' && key[2] == 'D' && key[3] == '_'
            //   prim->loops.attr_is<int>(key)) {
            //}

            auto &ch = channels.emplace_back();
            ch.numValues = uvsInd.size();
            ch.valueIndices = uvsInd.data();

            //void *chvp{};
            //prim->loops.attr_visit(key, [&] (auto const &arr) {
                //chvp = reinterpret_cast<void *>(arr.data());
            //});
            //assert(chvp);
            /*chanveckeys.push_back(key);*/
        /*}*/

        desc.numFVarChannels = channels.size();
        desc.fvarChannels = channels.data();
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
    if (!refiner) throw makeError("refiner is null (factory creation failed)");

    // Uniformly refine the topology up to 'maxlevel'
    // note: fullTopologyInLastLevel must be true to work with face-varying data
    {
        Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
        refineOptions.fullTopologyInLastLevel = hasLoopUVs;
        refiner->RefineUniform(refineOptions);
    }

    //// Allocate a buffer for vertex primvar data. The buffer length is set to
    //// be the sum of all children vertices up to the highest level of refinement.
    //std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    ////int nCoarseVerts = prim->verts.size();
    ////prim->verts.resize(refiner->GetNumVerticesTotal());
    ////Vertex * verts = reinterpret_cast<Vertex *>(prim->verts.data());
    //Vertex * verts = vbuffer.data();

    int nCoarseVerts = prim->verts.size();
    int nFineVerts   = refiner->GetLevel(maxlevel).GetNumVertices();
    int nTotalVerts  = refiner->GetNumVerticesTotal();
    int nTempVerts   = nTotalVerts - nCoarseVerts - nFineVerts;
    prim->verts.resize(nCoarseVerts + nTempVerts);

    AttrVector<vec2f> fine_uvs;
    int nCoarseFVars{}, nFineFVars{}, nTotalFVars{}, nTempFVars{};
    if (hasLoopUVs) {
        //for (int chi = 0; chi < channels.size(); chi++) {
            nCoarseFVars = prim->uvs.size(); //channels[0].numValues;
            nFineFVars = refiner->GetLevel(maxlevel).GetNumFVarValues();
            nTotalFVars = refiner->GetNumFVarValuesTotal();
            nTempFVars   = nTotalFVars - nCoarseFVars - nFineFVars;
            prim->uvs.resize(nCoarseFVars + nTempFVars);
            fine_uvs.resize(nFineFVars);
        //}
        //prim->loops.resize
    }

    //std::vector<Vertex> coarsePosBuffer(nCoarseVerts);
    //std::vector<Vertex> coarseClrBuffer(nCoarseVerts);

    // Initialize coarse mesh positions
    //{
        //auto &posarr = prim->verts.values;
        //auto &clrarr = prim->verts.add_attr<vec3f>("clr");
        //for (int i=0; i<nCoarseVerts; ++i) {
            //coarsePosBuffer[i].SetPoint(posarr[i][0], posarr[i][1], posarr[i][2]);
            //coarseClrBuffer[i].SetPoint(clrarr[i][0], clrarr[i][1], clrarr[i][2]);
        //}
    //}
    //AttrVector<vec3f> temp_verts(nTempVerts);
    AttrVector<vec3f> fine_verts(nFineVerts);

    //auto srcPos = reinterpret_cast<Vertex *>(prim->verts.data());
    //auto dstPos = srcPos + 1;
    //auto coarseClrBuffer = reinterpret_cast<Vertex const *>(prim->verts.attr<vec3f>("clr").data());

    //std::map<std::string, std::pair<void *, void *>> srcDstAttrs;
    //prim->verts.foreach_attr([&] (auto const &key, auto &arr) {
        //using T = std::decay_t<decltype(arr[0])>;
        //[>auto &temp_arr = <]temp_verts.add_attr<T>(key);
        ////srcDstAttrs[key] = {
            ////reinterpret_cast<void *>(arr.data()),
            ////reinterpret_cast<void *>(temp_arr.data()),
        ////};
    //});

    //std::vector<Vertex> tempPosBuffer(nTempVerts);
    //std::vector<Vertex> finePosBuffer(nFineVerts);

    //std::vector<Vertex> tempClrBuffer(nTempVerts);
    //std::vector<Vertex> fineClrBuffer(nFineVerts);


    // Interpolate vertex primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    //Vertex * src = verts;
    //Vertex * srcPos = &coarsePosBuffer[0];
    //Vertex * dstPos = &tempPosBuffer[0];

    //Vertex * srcClr = &coarseClrBuffer[0];
    //Vertex * dstClr = &tempClrBuffer[0];

    size_t srcposoffs = 0;
    size_t dstposoffs = nCoarseVerts;

    size_t srcfvaroffs{};
    size_t  dstfvaroffs{};
    if (hasLoopUVs) {
        dstfvaroffs = nCoarseFVars;
        //srcfvaroffs.resize(channels.size());
        //dstfvaroffs.resize(channels.size());
        //for (int i = 0; i < channels.size(); i++) {
            //dstfvaroffs[i] += channels[i].numValues;
        //}
    }

    for (int level = 1; level < maxlevel; ++level) {
        //Vertex * dst = src + refiner->GetLevel(level-1).GetNumVertices();
        //primvarRefiner.Interpolate(level, src, dst);
        //src = dst;
        auto *srcPos = convvertexptr(prim->verts.data() + srcposoffs);
        auto *dstPos = convvertexptr(prim->verts.data() + dstposoffs);
        primvarRefiner.Interpolate(       level, srcPos, dstPos);
        prim->verts.foreach_attr([&] (auto const &key, auto &arr) {
            auto *srcClr = convvertexptr(arr.data() + srcposoffs);
            auto *dstClr = convvertexptr(arr.data() + dstposoffs);
            primvarRefiner.InterpolateVarying(level, srcClr, dstClr);
        });
        if (hasLoopUVs) {
            //for (int chi = 0; chi < channels.size(); chi++) {
                //prim->loops.attr_visit(chanveckeys[chi], [&] (auto &chva) {
                    auto *srcFVarColor = convvertexptr(prim->uvs.data() + srcfvaroffs);
                    auto *dstFVarColor = convvertexptr(prim->uvs.data() + dstfvaroffs);
                    primvarRefiner.InterpolateFaceVarying(level, srcFVarColor, dstFVarColor);
                    auto numfvars = refiner->GetLevel(level).GetNumFVarValues();
                    srcfvaroffs = dstfvaroffs;
                    dstfvaroffs += numfvars;
                //});
            //}
        }
        //for (auto const &[key, arr]: srcDstAttrs) {
        //}
        auto numverts = refiner->GetLevel(level).GetNumVertices();
        srcposoffs = dstposoffs;
        dstposoffs += numverts;

        //srcPos = dstPos, dstPos += numverts;
        //srcClr = dstClr, dstClr += numverts;
    }

    // Interpolate the last level into the separate buffers for our final data:
    //primvarRefiner.Interpolate(       maxlevel, srcPos, finePosBuffer);
    //primvarRefiner.InterpolateVarying(maxlevel, srcClr, fineClrBuffer);
    {
        auto *srcPos = convvertexptr(prim->verts.data() + srcposoffs);
        auto *dstPos = convvertexptr(fine_verts.data());
        primvarRefiner.Interpolate(       maxlevel, srcPos, dstPos);
        prim->verts.foreach_attr([&] (auto const &key, auto &arr) {
            using T = std::decay_t<decltype(arr[0])>;
            auto &fine_arr = fine_verts.add_attr<T>(key);
            auto *srcClr = convvertexptr(arr.data() + srcposoffs);
            auto *dstClr = convvertexptr(fine_arr.data());
            primvarRefiner.InterpolateVarying(maxlevel, srcClr, dstClr);
        });
        if (hasLoopUVs) {
            //primvarRefiner.InterpolateFaceVarying(maxlevel, srcFVarColor, dstFVarColor, channelColor);
            //for (int chi = 0; chi < channels.size(); chi++) {
                //prim->loops.attr_visit(chanveckeys[chi], [&] (auto &chva) {
                    auto *srcFVarColor = convvertexptr(prim->uvs.data() + srcfvaroffs);
                    auto *dstFVarColor = convvertexptr(fine_uvs.data());
                    primvarRefiner.InterpolateFaceVarying(maxlevel, srcFVarColor, dstFVarColor);
                //});
            //}
        }
    }


    { // Output OBJ of the highest level refined -----------

        Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

        int nverts = refLastLevel.GetNumVertices();
        int nfaces = refLastLevel.GetNumFaces();

        //std::vector<int> nfvverts;
        int nfvars{};
        if (hasLoopUVs) {
            //nfvverts.resize(channels.size());
            //for (int i = 0; i < channels.size(); i++) {
                //nfvverts[i] = refLastLevel.GetNumFVarValues(i);
            //}
                nfvars = refLastLevel.GetNumFVarValues();
        }

        // Print vertex positions
        //int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;
        //assert(firstOfLastVerts == nCoarseVerts);
        //prim->verts->erase(prim->verts.begin(), prim->verts.begin() + firstOfLastVerts);

        std::swap(prim->verts, fine_verts);
        fine_verts.clear();
        fine_verts.shrink_to_fit();
        assert(prim->verts.size() == nverts);
        //prim->verts.resize(nverts);
        //for (int vert = 0; vert < nverts; ++vert) {
            //float const * pos = finePosBuffer[vert].GetPoint();
            ////printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
            //prim->verts[vert] = {pos[0], pos[1], pos[2]};
        //}

        std::swap(prim->uvs, fine_uvs);
        fine_uvs.clear();
        fine_uvs.shrink_to_fit();
        assert(prim->uvs.size() == nfvars);

        //{
            //auto &clrarr = prim->verts.add_attr<vec3f>("clr");
            //for (int i=0; i<nverts; ++i) {
                //float const * clr = fineClrBuffer[i].GetPoint();
                //clrarr[i] = {clr[0], clr[1], clr[2]};
            //}
        //}

    
        //prim->tris.clear();
        //prim->quads.clear();
        //prim->polys.clear();
        //prim->loops.clear();
        // Print faces
        if (triangulate) {
            prim->tris.resize(nfaces * 2);
            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4);

                auto &reftri1 = prim->tris[face * 2];
                auto &reftri2 = prim->tris[face * 2 + 1];
                reftri1[0] = fverts[0];
                reftri1[1] = fverts[1];
                reftri1[2] = fverts[2];
                reftri2[0] = fverts[0];
                reftri2[1] = fverts[2];
                reftri2[2] = fverts[3];

                //printf("f ");
                //for (int vert=0; vert<fverts.size(); ++vert) {
                    //printf("%d ", fverts[vert]+1); // OBJ uses 1-based arrays...
                //}
                //printf("\n");
            }

            if (hasLoopUVs) {  // very qianqiang uv0~2 for quads/tris, avoid use
                auto &uv0 = prim->tris.add_attr<vec3f>("uv0");
                auto &uv1 = prim->tris.add_attr<vec3f>("uv1");
                auto &uv2 = prim->tris.add_attr<vec3f>("uv2");
                for (int face = 0; face < nfaces; ++face) {
                    Far::ConstIndexArray fvars = refLastLevel.GetFaceFVarValues(face);
                    assert(fvars.size() == 4);
                    uv0[face*2] = v2to3(prim->uvs[fvars[0]]);
                    uv1[face*2] = v2to3(prim->uvs[fvars[1]]);
                    uv2[face*2] = v2to3(prim->uvs[fvars[2]]);
                    uv0[face*2+1] = v2to3(prim->uvs[fvars[0]]);
                    uv1[face*2+1] = v2to3(prim->uvs[fvars[2]]);
                    uv2[face*2+1] = v2to3(prim->uvs[fvars[3]]);
                }
                prim->uvs.clear();
            }

        } else if (asQuadFaces) {

            prim->quads.resize(nfaces);
            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4);

                auto &refquad = prim->quads[face];
                refquad[0] = fverts[0];
                refquad[1] = fverts[1];
                refquad[2] = fverts[2];
                refquad[3] = fverts[3];

                //printf("f ");
                //for (int vert=0; vert<fverts.size(); ++vert) {
                    //printf("%d ", fverts[vert]+1); // OBJ uses 1-based arrays...
                //}
                //printf("\n");
            }

            if (hasLoopUVs) {  // very qianqiang uv0~3 for quads/tris, avoid use
                auto &uv0 = prim->quads.add_attr<vec3f>("uv0");
                auto &uv1 = prim->quads.add_attr<vec3f>("uv1");
                auto &uv2 = prim->quads.add_attr<vec3f>("uv2");
                auto &uv3 = prim->quads.add_attr<vec3f>("uv3");
                for (int face = 0; face < nfaces; ++face) {
                    Far::ConstIndexArray fvars = refLastLevel.GetFaceFVarValues(face);
                    assert(fvars.size() == 4);
                    uv0[face] = v2to3(prim->uvs[fvars[0]]);
                    uv1[face] = v2to3(prim->uvs[fvars[1]]);
                    uv2[face] = v2to3(prim->uvs[fvars[2]]);
                    uv3[face] = v2to3(prim->uvs[fvars[3]]);
                }
                prim->uvs.clear();
            }

        } else {

            prim->polys.resize(nfaces);
            prim->loops.resize(nfaces * 4);

            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4);

                prim->loops[face*4+0] = fverts[0];
                prim->loops[face*4+1] = fverts[1];
                prim->loops[face*4+2] = fverts[2];
                prim->loops[face*4+3] = fverts[3];
                prim->polys[face] = {face * 4, 4};
            }

            if (hasLoopUVs) {
                auto &loop_uvs = prim->loops.attr<int>("uvs");
                loop_uvs.resize(nfaces * 4);

                for (int face = 0; face < nfaces; ++face) {
                    Far::ConstIndexArray fvars = refLastLevel.GetFaceFVarValues(face);
                    assert(fvars.size()==4);
                    loop_uvs[face*4+0] = fvars[0];
                    loop_uvs[face*4+1] = fvars[1];
                    loop_uvs[face*4+2] = fvars[2];
                    loop_uvs[face*4+3] = fvars[3];
                }
            }

        }

    }

        //refinedVerts += nCoarseVerts + ncfaces[0] + ncedges[0] + ncfaces[1];
        //nRefinedVerts = ncedges[1];

        //prim->verts.values.assign(refinedVerts, refinedVerts + nRefinedVerts);

        //prim->tris.clear();
        //prim->quads.clear();
        //prim->polys.clear();

        //printf("particle ");
        //for (int i=0; i<nRefinedVerts; ++i) {
            //float const * vert = refinedVerts + 3*i;
            //printf("-p %f %f %f\n", vert[0], vert[1], vert[2]);
        //}
        //printf("-c 1;\n");

    //delete stencilTable;
    //delete vbuffer;
}

struct OSDPrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int levels = get_input2<int>("levels");
        auto edgeCreaseAttr = get_input2<std::string>("edgeCreaseAttr");
        bool triangulate = get_input2<bool>("triangulate");
        bool asQuadFaces = get_input2<bool>("asQuadFaces");
        bool hasLoopUVs = get_input2<bool>("hasLoopUVs");
        if (levels) osdPrimSubdiv(prim.get(), levels, edgeCreaseAttr, triangulate,
                                  asQuadFaces, hasLoopUVs);
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDPrimSubdiv)({
    {
        "prim",
        {"int", "levels", "2"},
        {"string", "edgeCreaseAttr", ""},
        {"bool", "triangulate", "1"},
        {"bool", "asQuadFaces", "1"},
        {"bool", "hasLoopUVs", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});


//------------------------------------------------------------------------------

}
}
