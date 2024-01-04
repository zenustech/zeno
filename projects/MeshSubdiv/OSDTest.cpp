#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/zeno.h>
#include <cstdio>
#include <cstring>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>

namespace zeno {
namespace {
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

static float g_verts[8][3] = {{ -0.5f, -0.5f,  0.5f },
                              {  0.5f, -0.5f,  0.5f },
                              { -0.5f,  0.5f,  0.5f },
                              {  0.5f,  0.5f,  0.5f },
                              { -0.5f,  0.5f, -0.5f },
                              {  0.5f,  0.5f, -0.5f },
                              { -0.5f, -0.5f, -0.5f },
                              {  0.5f, -0.5f, -0.5f }};

static int g_nverts = 8,
           g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[24] = { 0, 1, 3, 2,
                                 2, 3, 5, 4,
                                 4, 5, 7, 6,
                                 6, 7, 1, 0,
                                 1, 7, 5, 3,
                                 6, 0, 2, 4  };

struct Coord3 {
    Coord3() { }
    Coord3(float x, float y, float z) { _xyz[0] = x, _xyz[1] = y, _xyz[2] = z; }

    void Clear() { _xyz[0] = _xyz[1] = _xyz[2] = 0.0f; }

    void AddWithWeight(Coord3 const & src, float weight) {
        _xyz[0] += weight * src._xyz[0];
        _xyz[1] += weight * src._xyz[1];
        _xyz[2] += weight * src._xyz[2];
    }

    float const * Coords() const { return &_xyz[0]; }

private:
    float _xyz[3];
};

struct Coord2 {
    Coord2() { }
    Coord2(float u, float v) { _uv[0] = u, _uv[1] = v; }

    void Clear() { _uv[0] = _uv[1] = 0.0f; }

    void AddWithWeight(Coord2 const & src, float weight) {
        _uv[0] += weight * src._uv[0];
        _uv[1] += weight * src._uv[1];
    }

    float const * Coords() const { return &_uv[0]; }

private:
    float _uv[2];
};

struct CoordBuffer {
    //
    //  The head of an external buffer and stride is specified on construction:
    //
    CoordBuffer(float * data, int size) : _data(data), _size(size) { }
    CoordBuffer() : _data(0), _size(0) { }

    void Clear() {
        for (int i = 0; i < _size; ++i) {
            _data[i] = 0.0f;
        }
    }

    void AddWithWeight(CoordBuffer const & src, float weight) {
        assert(src._size == _size);
        for (int i = 0; i < _size; ++i) {
            _data[i] += weight * src._data[i];
        }
    }

    float const * Coords() const { return _data; }

    //
    //  Defining [] to return a location elsewhere in the buffer is the key
    //  requirement to supporting interpolatible data of varying size
    //
    CoordBuffer operator[](int index) const {
        return CoordBuffer(_data + index * _size, _size);
    }

private:
    float * _data;
    int     _size;
};

using namespace OpenSubdiv;

struct OSDTest : INode {
    virtual void apply() override {
        // Populate a topology descriptor with our raw data

        typedef Far::TopologyDescriptor Descriptor;

        Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

        Sdc::Options options;
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

        Descriptor desc;
        desc.numVertices  = g_nverts;
        desc.numFaces     = g_nfaces;
        desc.numVertsPerFace = g_vertsperface;
        desc.vertIndicesPerFace  = g_vertIndices;


        // Instantiate a Far::TopologyRefiner from the descriptor
        Far::TopologyRefiner * refiner = Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                                                Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

        int maxlevel = get_input2<int>("maxLevel");

        // Uniformly refine the topology up to 'maxlevel'
        refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));


        // Allocate a buffer for vertex primvar data. The buffer length is set to
        // be the sum of all children vertices up to the highest level of refinement.
        zeno::log_info("refiner->GetNumVerticesTotal: {}", refiner->GetNumVerticesTotal());
        std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
        Vertex * verts = &vbuffer[0];


        // Initialize coarse mesh positions
        int nCoarseVerts = g_nverts;
        for (int i=0; i<nCoarseVerts; ++i) {
            verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
        }


        // Interpolate vertex primvar data
        Far::PrimvarRefiner primvarRefiner(*refiner);

        Vertex * src = verts;
        for (int level = 1; level <= maxlevel; ++level) {
            Vertex * dst = src + refiner->GetLevel(level-1).GetNumVertices();
            primvarRefiner.Interpolate(level, src, dst);
            src = dst;
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<int> loops;
        std::vector<int> polys;
        { // Output OBJ of the highest level refined -----------

            Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

            int nverts = refLastLevel.GetNumVertices();
            int nfaces = refLastLevel.GetNumFaces();

            // Print vertex positions
            int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;
            prim->verts.resize(nverts);

            for (int vert = 0; vert < nverts; ++vert) {
                float const * pos = verts[firstOfLastVerts + vert].GetPosition();
                prim->verts[vert] = {pos[0], pos[1], pos[2]};
            }


            // Print faces
            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4);
                polys.push_back(fverts.size());

                for (int vert=0; vert<fverts.size(); ++vert) {
                    loops.push_back(fverts[vert]);
                }
            }
        }
        prim->loops.resize(loops.size());
        std::copy(loops.begin(), loops.end(), prim->loops.begin());
        prim->polys.resize(polys.size());
        int start = 0;
        for (auto i = 0; i < prim->polys.size(); i++) {
            prim->polys[i] = {start, polys[i]};
            start += polys[i];
        }

        delete refiner;

        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDTest)
({
    {
        {"int", "maxLevel", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});
static Far::TopologyRefiner *
createFarTopologyRefiner() {

    typedef Far::TopologyDescriptor Descriptor;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    Descriptor desc;
    desc.numVertices  = g_nverts;
    desc.numFaces     = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace  = g_vertIndices;

    // Instantiate a Far::TopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                    Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    return refiner;
}

struct OSDTest2 : INode {
    virtual void apply() override {
        //  Instantiate a Far::TopologyRefiner from the global geometry:
        Far::TopologyRefiner * refiner = createFarTopologyRefiner();

        //  Uniformly refine the topology up to 'maxlevel'
        int maxlevel = get_input2<int>("maxLevel");

        refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

        //  Allocate and populate data buffers for vertex primvar data -- positions and
        //  UVs. We assign UV coordiantes by simply projecting/assigning XY values.
        //  The position and UV buffers use their associated data types, while the
        //  combined buffer uses 5 floats per vertex.
        //
        int numBaseVertices  = g_nverts;
        int numTotalVertices = refiner->GetNumVerticesTotal();

        std::vector<Coord3> posData(numTotalVertices);
        std::vector<Coord2> uvData(numTotalVertices);

        int                 combinedStride = 3 + 2;
        std::vector<float>  combinedData(numTotalVertices * combinedStride);

        for (int i = 0; i < numBaseVertices; ++i) {
            posData[i] = Coord3(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
            uvData[i]  = Coord2(g_verts[i][0], g_verts[i][1]);

            float * coordCombined = &combinedData[i * combinedStride];
            coordCombined[0] = g_verts[i][0];
            coordCombined[1] = g_verts[i][1];
            coordCombined[2] = g_verts[i][2];
            coordCombined[3] = g_verts[i][0];
            coordCombined[4] = g_verts[i][1];
        }

        //  Interpolate vertex primvar data
        Far::PrimvarRefiner primvarRefiner(*refiner);

        Coord3 * posSrc = &posData[0];
        Coord2 * uvSrc  = & uvData[0];

        CoordBuffer combinedSrc(&combinedData[0], combinedStride);

        for (int level = 1; level <= maxlevel; ++level) {
            int numLevelVerts = refiner->GetLevel(level-1).GetNumVertices();

            Coord3 * posDst = posSrc + numLevelVerts;
            Coord2 * uvDst  = uvSrc + numLevelVerts;

            CoordBuffer combinedDst = combinedSrc[numLevelVerts];

            primvarRefiner.Interpolate(level, posSrc, posDst);
            primvarRefiner.Interpolate(level, uvSrc, uvDst);
            primvarRefiner.Interpolate(level, combinedSrc, combinedDst);

            posSrc = posDst;
            uvSrc = uvDst;
            combinedSrc = combinedDst;
        }


        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<vec3f> verts;
        std::vector<vec2f> uvs;
        std::vector<int> loops;
        std::vector<int> polys;
        //
        //  Output OBJ of the highest level refined:
        //
        Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

        int firstOfLastVerts = numTotalVertices - refLastLevel.GetNumVertices();

        //  Print vertex positions
        for (int vert = firstOfLastVerts; vert < numTotalVertices; ++vert) {
            float const * pos = &combinedData[vert * combinedStride];
            verts.emplace_back(pos[0], pos[1], pos[2]);
        }

        printf("#  UV coordinates:\n");
        for (int vert = firstOfLastVerts; vert < numTotalVertices; ++vert) {
            float const * uv = &combinedData[vert * combinedStride] + 3;
            uvs.emplace_back(uv[0], uv[1]);
        }

        //  Print faces
        int numFaces = refLastLevel.GetNumFaces();

        for (int face = 0; face < numFaces; ++face) {
            Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

            polys.push_back(fverts.size());
            for (int fvert = 0; fvert < fverts.size(); ++fvert) {
                loops.push_back(fverts[fvert]);
            }
        }

        delete refiner;

        prim->verts.resize(verts.size());
        std::copy(verts.begin(), verts.end(), prim->verts.begin());
        prim->uvs.resize(verts.size());
        std::copy(uvs.begin(), uvs.end(), prim->uvs.begin());
        prim->loops.resize(loops.size());
        std::copy(loops.begin(), loops.end(), prim->loops.begin());
        std::copy(loops.begin(), loops.end(), prim->loops.add_attr<int>("uvs").begin());

        prim->polys.resize(polys.size());
        int start = 0;
        for (auto i = 0; i < prim->polys.size(); i++) {
            prim->polys[i] = {start, polys[i]};
            start += polys[i];
        }
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDTest2)
({
    {
        {"int", "maxLevel", "1"},
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
