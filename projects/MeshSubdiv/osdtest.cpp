#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/types/PrimitiveObject.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>
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
static void osdPrimSubdiv(PrimitiveObject *prim, int levels) {

    int maxlevel=levels,
        nCoarseVerts=0,
        nRefinedVerts=0;
    std::vector<int> ncfaces(maxlevel);
    std::vector<int> ncedges(maxlevel);

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

    int offsetred = polysInd.size();
    polysLen.resize(prim->tris.size() + prim->quads.size() + prim->polys.size());
    polysInd.resize(offsetred + primpolyreduced);
    for (int i = 0; i < prim->polys.size(); i++) {
        auto [base, len] = prim->polys[i];
        polysLen[prim->tris.size() + prim->quads.size() + i] = len;
        for (int j = 0; j < len; j++) {
            polysInd[offsetred + j] = prim->loops[base + j];
        }
        offsetred += len;
    }
    //ZENO_P(polysLen.size());
    //ZENO_P(polysInd.size());

    //
    // Setup phase
    //
    Far::StencilTable const * stencilTable = NULL;
    { // Setup Far::StencilTable

        //---begin Far::TopologyRefiner const * refiner = createTopologyRefiner(maxlevel);
        Far::TopologyRefiner * refiner;
        {
            Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

            Sdc::Options options;
            options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

            Far::TopologyDescriptor desc;
            desc.numVertices = prim->verts.size();
            desc.numFaces = polysLen.size();
            desc.numVertsPerFace = polysLen.data();
            desc.vertIndicesPerFace = polysInd.data();

            // Instantiate a FarTopologyRefiner from the descriptor
            refiner = Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Create(desc,
                    Far::TopologyRefinerFactory<Far::TopologyDescriptor>::Options(type, options));

            // Uniformly refine the topology up to 'maxlevel'
            refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));
        }//---end Far::TopologyRefiner const * refiner = createTopologyRefiner(maxlevel);

        // Setup a factory to create FarStencilTable (for more details see
        // Far tutorials)
        Far::StencilTableFactory::Options options;
        options.generateOffsets=true;
        options.generateIntermediateLevels=false;
        //options.maxLevel = 10;

        stencilTable = Far::StencilTableFactory::Create(*refiner, options);

        nCoarseVerts = refiner->GetLevel(0).GetNumVertices();
        for (int l = 0; l < maxlevel; l++) {
            ncfaces[l] = refiner->GetLevel(l).GetNumFaces();
            ncedges[l] = refiner->GetLevel(l).GetNumEdges();
        }
        nRefinedVerts = stencilTable->GetNumStencils();
        log_info("[osd] coarse verts: {}", nCoarseVerts);
        log_info("[osd] refined verts: {}", nRefinedVerts);

        // We are done with Far: cleanup table
        delete refiner;
        polysInd.clear();
        polysInd.shrink_to_fit();
        polysLen.clear();
        polysLen.shrink_to_fit();
    }

    // Setup a buffer for vertex primvar data:
    Osd::CpuVertexBuffer * vbuffer =
        Osd::CpuVertexBuffer::Create(3, nCoarseVerts + nRefinedVerts);

    //
    // Execution phase (every frame)
    //
    {
        // Pack the control vertex data at the start of the vertex buffer
        // and update every time control data changes
        vbuffer->UpdateData(reinterpret_cast<float const *>(prim->verts.data()), 0, nCoarseVerts);


        Osd::BufferDescriptor srcDesc(0, 3, 3);
        Osd::BufferDescriptor dstDesc(nCoarseVerts*3, 3, 3);

        // Launch the computation
        Osd::CpuEvaluator::EvalStencils(vbuffer, srcDesc,
                                        vbuffer, dstDesc,
                                        stencilTable);
    }

    { // Visualization with Maya : print a MEL script that generates particles
      // at the location of the refined vertices

        vec3f const * refinedVerts = reinterpret_cast<vec3f const *>(
            vbuffer->BindCpuBuffer()) + nCoarseVerts;

        //refinedVerts += nCoarseVerts + ncfaces[0] + ncedges[0] + ncfaces[1];
        //nRefinedVerts = ncedges[1];

        prim->verts.values.assign(refinedVerts, refinedVerts + nRefinedVerts);

        prim->tris.clear();
        prim->quads.clear();
        prim->polys.clear();

        //printf("particle ");
        //for (int i=0; i<nRefinedVerts; ++i) {
            //float const * vert = refinedVerts + 3*i;
            //printf("-p %f %f %f\n", vert[0], vert[1], vert[2]);
        //}
        //printf("-c 1;\n");
    }

    delete stencilTable;
    delete vbuffer;
}

struct OSDPrimSubdiv : INode {
    virtual void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int levels = get_input2<int>("levels");
        if (levels) osdPrimSubdiv(prim.get(), levels);
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDPrimSubdiv)({
    {
        "prim",
        {"int", "levels", "2"},
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
