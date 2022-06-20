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


//------------------------------------------------------------------------------
static void osdPrimSubdiv(PrimitiveObject *prim, int levels, bool triangulate = false) {

    const int maxlevel=levels;
    if (maxlevel <= 0 || !prim->verts.size()) return;

        //nCoarseVerts=0,
        //nRefinedVerts=0;
    //std::vector<int> ncfaces(maxlevel);
    //std::vector<int> ncedges(maxlevel);

    std::vector<int> polysInd, polysLen;
    int primpolyreduced = 0;
    for (int i = 0; i < prim->polys.size(); i++) {
        primpolyreduced += prim->polys[i].second;
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
    
    prim->tris.clear();
    prim->quads.clear();
    prim->polys.clear();
    prim->loops.clear();

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Far::TopologyDescriptor desc;
            desc.numVertices = prim->verts.size();
            desc.numFaces = polysLen.size();
            desc.numVertsPerFace = polysLen.data();
            desc.vertIndicesPerFace = polysInd.data();


    // Instantiate a Far::TopologyRefiner from the descriptor
            using Factory = Far::TopologyRefinerFactory<Far::TopologyDescriptor>;
    Far::TopologyRefiner * refiner = Factory::Create(desc, Factory::Options(type, options));
    if (!refiner) throw makeError("refiner is null");

    // Uniformly refine the topology up to 'maxlevel'
    refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

    struct Vertex {

        // Minimal required interface ----------------------
        Vertex() { }

        Vertex(Vertex const & src) {
            _position[0] = src._position[0];
            _position[1] = src._position[1];
            _position[2] = src._position[2];
        }

        void Clear( void * =0 ) {
            _position[0]=_position[1]=_position[2]=0.0f;
        }

        void AddWithWeight(Vertex const & src, float weight) {
            _position[0]+=weight*src._position[0];
            _position[1]+=weight*src._position[1];
            _position[2]+=weight*src._position[2];
        }

        // Public interface ------------------------------------
        void SetPosition(float x, float y, float z) {
            _position[0]=x;
            _position[1]=y;
            _position[2]=z;
        }

        const float * GetPosition() const {
            return _position;
        }

    private:
        float _position[3];
    };

    // Allocate a buffer for vertex primvar data. The buffer length is set to
    // be the sum of all children vertices up to the highest level of refinement.
    std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
    //int nCoarseVerts = prim->verts.size();
    //prim->verts.resize(refiner->GetNumVerticesTotal());
    //Vertex * verts = reinterpret_cast<Vertex *>(prim->verts.data());
    Vertex * verts = vbuffer.data();


    // Initialize coarse mesh positions
    int nCoarseVerts = prim->verts.size();
    for (int i=0; i<nCoarseVerts; ++i) {
        verts[i].SetPosition(prim->verts[i][0], prim->verts[i][1], prim->verts[i][2]);
    }


    // Interpolate vertex primvar data
    Far::PrimvarRefiner primvarRefiner(*refiner);

    Vertex * src = verts;
    for (int level = 1; level <= maxlevel; ++level) {
        Vertex * dst = src + refiner->GetLevel(level-1).GetNumVertices();
        primvarRefiner.Interpolate(level, src, dst);
        src = dst;
    }


    { // Output OBJ of the highest level refined -----------

        Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

        int nverts = refLastLevel.GetNumVertices();
        int nfaces = refLastLevel.GetNumFaces();

        // Print vertex positions
        int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;
        //assert(firstOfLastVerts == nCoarseVerts);
        //prim->verts->erase(prim->verts.begin(), prim->verts.begin() + firstOfLastVerts);

        prim->verts.resize(nverts);
        for (int vert = 0; vert < nverts; ++vert) {
            float const * pos = verts[firstOfLastVerts + vert].GetPosition();
            //printf("v %f %f %f\n", pos[0], pos[1], pos[2]);
            prim->verts[vert] = {pos[0], pos[1], pos[2]};
        }

        // Print faces
        if (!triangulate) {
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
        } else {
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
    //}

    //delete stencilTable;
    //delete vbuffer;
}

struct OSDPrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int levels = get_input2<int>("levels");
        bool triangulate = get_input2<bool>("triangulate");
        if (levels) osdPrimSubdiv(prim.get(), levels, 1);
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDPrimSubdiv)({
    {
        "prim",
        {"int", "levels", "2"},
        {"bool", "triangulate", "true"},
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
