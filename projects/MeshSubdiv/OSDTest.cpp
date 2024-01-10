#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>
#include <zeno/utils/log.h>
#include <zeno/utils/zeno_p.h>
#include <zeno/zeno.h>
#include <cstdio>
#include <cstring>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>
#include "zeno/funcs/PrimitiveUtils.h"

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

// Per-vertex RGB color data
static float g_colors[8][3] = {{ 1.0f, 0.0f, 0.5f },
                               { 0.0f, 1.0f, 0.0f },
                               { 0.0f, 0.0f, 1.0f },
                               { 1.0f, 1.0f, 1.0f },
                               { 1.0f, 1.0f, 0.0f },
                               { 0.0f, 1.0f, 1.0f },
                               { 1.0f, 0.0f, 1.0f },
                               { 0.0f, 0.0f, 0.0f }};
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

struct Point3 {

    // Minimal required interface ----------------------
    Point3() { }

    void Clear( void * =0 ) {
        _point[0]=_point[1]=_point[2]=0.0f;
    }

    void AddWithWeight(Point3 const & src, float weight) {
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

typedef Point3 VertexPosition;
typedef Point3 VertexColor;

struct OSDTest3 : INode {
    virtual void apply() override {
        int maxlevel = get_input2<int>("maxLevel");

        Far::TopologyRefiner * refiner = createFarTopologyRefiner();

        // Uniformly refine the topology up to 'maxlevel'
        refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxlevel));

        // Allocate buffers for vertex primvar data.
        //
        // We assume we received the coarse data for the mesh in separate buffers
        // from some other source, e.g. an Alembic file.  Meanwhile, we want buffers
        // for the last/finest subdivision level to persist.  We have no interest
        // in the intermediate levels.
        //
        // Determine the sizes for our needs:
        int nCoarseVerts = g_nverts;
        int nFineVerts   = refiner->GetLevel(maxlevel).GetNumVertices();
        int nTotalVerts  = refiner->GetNumVerticesTotal();
        int nTempVerts   = nTotalVerts - nCoarseVerts - nFineVerts;

        // Allocate and initialize the primvar data for the original coarse vertices:
        std::vector<VertexPosition> coarsePosBuffer(nCoarseVerts);
        std::vector<VertexColor>    coarseClrBuffer(nCoarseVerts);

        for (int i = 0; i < nCoarseVerts; ++i) {
            coarsePosBuffer[i].SetPoint(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
            coarseClrBuffer[i].SetPoint(g_colors[i][0], g_colors[i][1], g_colors[i][2]);
        }

        // Allocate intermediate and final storage to be populated:
        std::vector<VertexPosition> tempPosBuffer(nTempVerts);
        std::vector<VertexPosition> finePosBuffer(nFineVerts);

        std::vector<VertexColor> tempClrBuffer(nTempVerts);
        std::vector<VertexColor> fineClrBuffer(nFineVerts);

        // Interpolate all primvar data -- separate buffers can be populated on
        // separate threads if desired:
        VertexPosition * srcPos = &coarsePosBuffer[0];
        VertexPosition * dstPos = &tempPosBuffer[0];

        VertexColor * srcClr = &coarseClrBuffer[0];
        VertexColor * dstClr = &tempClrBuffer[0];

        Far::PrimvarRefiner primvarRefiner(*refiner);

        for (int level = 1; level < maxlevel; ++level) {
            primvarRefiner.Interpolate(       level, srcPos, dstPos);
            primvarRefiner.InterpolateVarying(level, srcClr, dstClr);

            srcPos = dstPos, dstPos += refiner->GetLevel(level).GetNumVertices();
            srcClr = dstClr, dstClr += refiner->GetLevel(level).GetNumVertices();
        }

        // Interpolate the last level into the separate buffers for our final data:
        primvarRefiner.Interpolate(       maxlevel, srcPos, finePosBuffer);
        primvarRefiner.InterpolateVarying(maxlevel, srcClr, fineClrBuffer);

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<vec3f> verts;
        std::vector<vec3f> clrs;
        std::vector<vec2f> uvs;
        std::vector<int> loops;
        std::vector<int> polys;
        { // Visualization with Maya : print a MEL script that generates colored
          // particles at the location of the refined vertices (don't forget to
          // turn shading on in the viewport to see the colors)

            int nverts = nFineVerts;

            // Output particle positions
            verts.resize(nverts);
            for (int vert = 0; vert < nverts; ++vert) {
                float const * pos = finePosBuffer[vert].GetPoint();
                verts[vert] = {pos[0], pos[1], pos[2]};
            }

            clrs.resize(nverts);
            // Set per-particle color values from our primvar data
            for (int vert = 0; vert < nverts; ++vert) {
                float const * color = fineClrBuffer[vert].GetPoint();
                clrs[vert] = {color[0], color[1], color[2]};
            }
            printf(";\n");
        }

        delete refiner;


        prim->verts.resize(verts.size());
        std::copy(verts.begin(), verts.end(), prim->verts.begin());
        auto &clr = prim->verts.add_attr<vec3f>("clr");
        std::copy(clrs.begin(), clrs.end(), clr.begin());
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
ZENO_DEFNODE(OSDTest3)
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
struct FVarVertexUV {

    // Minimal required interface ----------------------
    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertexUV const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }

    // Basic 'uv' layout channel
    float u,v;
};

struct FVarVertexColor {

    // Minimal required interface ----------------------
    void Clear() {
        r=g=b=a=0.0f;
    }

    void AddWithWeight(FVarVertexColor const & src, float weight) {
        r += weight * src.r;
        g += weight * src.g;
        b += weight * src.b;
        a += weight * src.a;
    }

    // Basic 'color' layout channel
    float r,g,b,a;
};
// 'face-varying' primitive variable data & topology for UVs
static float g_uvs[14][2] = {{ 0.375, 0.00 },
                             { 0.625, 0.00 },
                             { 0.375, 0.25 },
                             { 0.625, 0.25 },
                             { 0.375, 0.50 },
                             { 0.625, 0.50 },
                             { 0.375, 0.75 },
                             { 0.625, 0.75 },
                             { 0.375, 1.00 },
                             { 0.625, 1.00 },
                             { 0.875, 0.00 },
                             { 0.875, 0.25 },
                             { 0.125, 0.00 },
                             { 0.125, 0.25 }};

static int g_nuvs = 14;

static int g_uvIndices[24] = {  0,  1,  3,  2,
                                2,  3,  5,  4,
                                4,  5,  7,  6,
                                6,  7,  9,  8,
                                1, 10, 11,  3,
                               12,  0,  2, 13  };
// 'face-varying' primitive variable data & topology for color
static float g_colors_[24][4] = {{1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 0.0, 0.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0},
                                {1.0, 1.0, 1.0, 1.0}};

static int g_ncolors = 24;

static int g_colorIndices[24] = { 0,  3,  9,  6,
                                  7, 10, 15, 12,
                                 13, 16, 21, 18,
                                 19, 22,  4,  1,
                                  5, 23, 17, 11,
                                 20,  2,  8, 14 };

struct OSDTest4 : INode {
    virtual void apply() override {
        int maxlevel = get_input2<int>("maxLevel");
        typedef Far::TopologyDescriptor Descriptor;

        Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

        Sdc::Options options;
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
        options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_NONE);

        // Populate a topology descriptor with our raw data
        Descriptor desc;
        desc.numVertices  = g_nverts;
        desc.numFaces     = g_nfaces;
        desc.numVertsPerFace = g_vertsperface;
        desc.vertIndicesPerFace  = g_vertIndices;

        int channelUV = 0;
        int channelColor = 1;

        // Create a face-varying channel descriptor
        Descriptor::FVarChannel channels[2];
        channels[channelUV].numValues = g_nuvs;
        channels[channelUV].valueIndices = g_uvIndices;
        channels[channelColor].numValues = g_ncolors;
        channels[channelColor].valueIndices = g_colorIndices;

        // Add the channel topology to the main descriptor
        desc.numFVarChannels = 2;
        desc.fvarChannels = channels;

        // Instantiate a Far::TopologyRefiner from the descriptor
        Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

        // Uniformly refine the topology up to 'maxlevel'
        // note: fullTopologyInLastLevel must be true to work with face-varying data
        {
            Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
            refineOptions.fullTopologyInLastLevel = true;
            refiner->RefineUniform(refineOptions);
        }

        // Allocate and initialize the 'vertex' primvar data (see tutorial 2 for
        // more details).
        std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
        Vertex * verts = &vbuffer[0];

        for (int i=0; i<g_nverts; ++i) {
            verts[i].SetPosition(g_verts[i][0], g_verts[i][1], g_verts[i][2]);
        }

        // Allocate and initialize the first channel of 'face-varying' primvar data (UVs)
        std::vector<FVarVertexUV> fvBufferUV(refiner->GetNumFVarValuesTotal(channelUV));
        FVarVertexUV * fvVertsUV = &fvBufferUV[0];
        for (int i=0; i<g_nuvs; ++i) {
            fvVertsUV[i].u = g_uvs[i][0];
            fvVertsUV[i].v = g_uvs[i][1];
        }

        // Allocate & interpolate the 'face-varying' primvar data (colors)
        std::vector<FVarVertexColor> fvBufferColor(refiner->GetNumFVarValuesTotal(channelColor));
        FVarVertexColor * fvVertsColor = &fvBufferColor[0];
        for (int i=0; i<g_ncolors; ++i) {
            fvVertsColor[i].r = g_colors_[i][0];
            fvVertsColor[i].g = g_colors_[i][1];
            fvVertsColor[i].b = g_colors_[i][2];
            fvVertsColor[i].a = g_colors_[i][3];
        }

        // Interpolate both vertex and face-varying primvar data
        Far::PrimvarRefiner primvarRefiner(*refiner);

        Vertex *     srcVert = verts;
        FVarVertexUV * srcFVarUV = fvVertsUV;
        FVarVertexColor * srcFVarColor = fvVertsColor;

        for (int level = 1; level <= maxlevel; ++level) {
            Vertex *     dstVert = srcVert + refiner->GetLevel(level-1).GetNumVertices();
            FVarVertexUV * dstFVarUV = srcFVarUV + refiner->GetLevel(level-1).GetNumFVarValues(channelUV);
            FVarVertexColor * dstFVarColor = srcFVarColor + refiner->GetLevel(level-1).GetNumFVarValues(channelColor);

            primvarRefiner.Interpolate(level, srcVert, dstVert);
            primvarRefiner.InterpolateFaceVarying(level, srcFVarUV, dstFVarUV, channelUV);
            primvarRefiner.InterpolateFaceVarying(level, srcFVarColor, dstFVarColor, channelColor);

            srcVert = dstVert;
            srcFVarUV = dstFVarUV;
            srcFVarColor = dstFVarColor;
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<vec3f> verts_;
        std::vector<vec2f> uvs;
        std::vector<int> loops;
        std::vector<int> loops_uv;
        std::vector<int> polys;
        { // Output OBJ of the highest level refined -----------

            Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

            int nverts = refLastLevel.GetNumVertices();
            zeno::log_info("nverts: {}", nverts);
            int nuvs   = refLastLevel.GetNumFVarValues(channelUV);
            zeno::log_info("nuvs: {}", nuvs);
            int ncolors= refLastLevel.GetNumFVarValues(channelColor);
            zeno::log_info("ncolors: {}", ncolors);
            int nfaces = refLastLevel.GetNumFaces();
            zeno::log_info("nfaces: {}", nfaces);

            // Print vertex positions
            int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

            for (int vert = 0; vert < nverts; ++vert) {
                float const * pos = verts[firstOfLastVerts + vert].GetPosition();
                verts_.emplace_back(pos[0], pos[1], pos[2]);
            }

            // Print uvs
            int firstOfLastUvs = refiner->GetNumFVarValuesTotal(channelUV) - nuvs;

            for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
                FVarVertexUV const & uv = fvVertsUV[firstOfLastUvs + fvvert];
                uvs.emplace_back(uv.u, uv.v);
            }

            // Print colors
            int firstOfLastColors = refiner->GetNumFVarValuesTotal(channelColor) - ncolors;

            for (int fvvert = 0; fvvert < ncolors; ++fvvert) {
                FVarVertexColor const & c = fvVertsColor[firstOfLastColors + fvvert];
//                printf("c %f %f %f %f\n", c.r, c.g, c.b, c.a);
            }

            // Print faces
            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);
                Far::ConstIndexArray fuvs   = refLastLevel.GetFaceFVarValues(face, channelUV);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4 && fuvs.size()==4);

                polys.push_back(fverts.size());
                for (int vert=0; vert<fverts.size(); ++vert) {
                    loops.push_back(fverts[vert]);
                    loops_uv.push_back(fuvs[vert]);
                }
            }
        }

        delete refiner;

        prim->verts.resize(verts_.size());
        std::copy(verts_.begin(), verts_.end(), prim->verts.begin());
        prim->uvs.resize(uvs.size());
        std::copy(uvs.begin(), uvs.end(), prim->uvs.begin());
        prim->loops.resize(loops.size());
        std::copy(loops.begin(), loops.end(), prim->loops.begin());
        std::copy(loops_uv.begin(), loops_uv.end(), prim->loops.add_attr<int>("uvs").begin());

        prim->polys.resize(polys.size());
        int start = 0;
        for (auto i = 0; i < prim->polys.size(); i++) {
            prim->polys[i] = {start, polys[i]};
            start += polys[i];
        }
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(OSDTest4)
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

struct PrimSubdivision : INode {
    virtual void apply() override {
        auto in_prim = get_input<PrimitiveObject>("prim");
        int maxlevel = get_input2<int>("maxLevel");
        if (in_prim->tris.size() || in_prim->lines.size()) {
            primPolygonate(in_prim.get(), true);
        }
        typedef Far::TopologyDescriptor Descriptor;

        Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

        Sdc::Options options;
        options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

        std::vector<int> polys_in;
        for (auto [_base, len]: in_prim->polys) {
            polys_in.push_back(len);
        }
        // Populate a topology descriptor with our raw data
        Descriptor desc;
        desc.numVertices  = in_prim->verts.size();
        desc.numFaces     = in_prim->polys.size();
        desc.numVertsPerFace = polys_in.data();
        desc.vertIndicesPerFace  = in_prim->loops.data();

        int channelUV = 0;

        if (in_prim->uvs.size()) {
            // Create a face-varying channel descriptor
            Descriptor::FVarChannel channels[1];
            channels[channelUV].numValues = in_prim->uvs.size();
            channels[channelUV].valueIndices = in_prim->loops.attr<int>("uvs").data();

            // Add the channel topology to the main descriptor
            desc.numFVarChannels = 1;
            desc.fvarChannels = channels;
        }

        // Instantiate a Far::TopologyRefiner from the descriptor
        Far::TopologyRefiner * refiner =
            Far::TopologyRefinerFactory<Descriptor>::Create(desc,
                Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

        // Uniformly refine the topology up to 'maxlevel'
        // note: fullTopologyInLastLevel must be true to work with face-varying data
        {
            Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
            refineOptions.fullTopologyInLastLevel = true;
            refiner->RefineUniform(refineOptions);
        }

        // Allocate and initialize the 'vertex' primvar data (see tutorial 2 for
        // more details).
        in_prim->verts.resize(refiner->GetNumVerticesTotal());

        // Interpolate both vertex and face-varying primvar data
        Far::PrimvarRefiner primvarRefiner(*refiner);

        int start_offset = 0;

        for (int level = 1; level <= maxlevel; ++level) {
            auto *     srcVert = convvertexptr(in_prim->verts.data() + start_offset);
            int end_offset = start_offset + refiner->GetLevel(level-1).GetNumVertices();
            auto *     dstVert = convvertexptr(in_prim->verts.data() + end_offset);

            primvarRefiner.Interpolate(level, srcVert, dstVert);
            in_prim->verts.foreach_attr([&](auto const &key, auto &arr) {
                auto *srcClr = convvertexptr(arr.data() + start_offset);
                auto *dstClr = convvertexptr(arr.data() + end_offset);
                primvarRefiner.InterpolateVarying(level, srcClr, dstClr);
            });
            start_offset = end_offset;
        }

        std::vector<FVarVertexUV> fvBufferUV;
        if (in_prim->uvs.size()) {
            // Allocate and initialize the first channel of 'face-varying' primvar data (UVs)
            fvBufferUV.resize(refiner->GetNumFVarValuesTotal(channelUV));
            for (int i=0; i<in_prim->uvs.size(); ++i) {
                fvBufferUV[i].u = in_prim->uvs[i][0];
                fvBufferUV[i].v = in_prim->uvs[i][1];
            }
            FVarVertexUV * srcFVarUV = fvBufferUV.data();
            for (int level = 1; level <= maxlevel; ++level) {
                FVarVertexUV * dstFVarUV = srcFVarUV + refiner->GetLevel(level-1).GetNumFVarValues(channelUV);

                primvarRefiner.InterpolateFaceVarying(level, srcFVarUV, dstFVarUV, channelUV);

                srcFVarUV = dstFVarUV;
            }
        }

        auto prim = std::make_shared<zeno::PrimitiveObject>();
        std::vector<vec2f> uvs;
        std::vector<int> loops;
        std::vector<int> loops_uv;
        std::vector<int> polys;
        { // Output OBJ of the highest level refined -----------

            Far::TopologyLevel const & refLastLevel = refiner->GetLevel(maxlevel);

            int nverts = refLastLevel.GetNumVertices();
            zeno::log_info("nverts: {}", nverts);

            int nfaces = refLastLevel.GetNumFaces();
            zeno::log_info("nfaces: {}", nfaces);

            // Print vertex positions
            int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

            prim->verts.resize(nverts);
            std::copy(in_prim->verts.begin() + firstOfLastVerts, in_prim->verts.end(), prim->verts.begin());
            in_prim->verts.foreach_attr([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &new_arr = prim->verts.template add_attr<T>(key);
                std::copy(arr.begin() + firstOfLastVerts, arr.end(), new_arr.begin());
            });

            int nuvs = 0;
            if (in_prim->uvs.size()) {
                int nuvs   = refLastLevel.GetNumFVarValues(channelUV);
                zeno::log_info("nuvs: {}", nuvs);
                // Print uvs
                int firstOfLastUvs = refiner->GetNumFVarValuesTotal(channelUV) - nuvs;

                for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
                    FVarVertexUV const & uv = fvBufferUV[firstOfLastUvs + fvvert];
                    uvs.emplace_back(uv.u, uv.v);
                }
            }

            // Print faces
            for (int face = 0; face < nfaces; ++face) {

                Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);

                // all refined Catmark faces should be quads
                assert(fverts.size()==4 && fuvs.size()==4);

                polys.push_back(fverts.size());
                for (int vert=0; vert<fverts.size(); ++vert) {
                    loops.push_back(fverts[vert]);
                }
                if (in_prim->uvs.size()) {
                    Far::ConstIndexArray fuvs   = refLastLevel.GetFaceFVarValues(face, channelUV);
                    for (int vert=0; vert<fverts.size(); ++vert) {
                        loops_uv.push_back(fuvs[vert]);
                    }
                }
            }
        }

        delete refiner;

        prim->loops.resize(loops.size());
        std::copy(loops.begin(), loops.end(), prim->loops.begin());
        if (uvs.size()) {
            prim->uvs.resize(uvs.size());
            std::copy(uvs.begin(), uvs.end(), prim->uvs.begin());
            std::copy(loops_uv.begin(), loops_uv.end(), prim->loops.add_attr<int>("uvs").begin());
        }

        prim->polys.resize(polys.size());
        int start = 0;
        for (auto i = 0; i < prim->polys.size(); i++) {
            prim->polys[i] = {start, polys[i]};
            start += polys[i];
        }
        if (get_input2<bool>("triangulate")) {
            primTriangulate(prim.get(), true, true);
        }
        set_output("prim", std::move(prim));
    }
};
ZENO_DEFNODE(PrimSubdivision)
({
    {
        "prim",
        {"int", "maxLevel", "1"},
        {"bool", "triangulate", "1"},
    },
    {
        "prim",
    },
    {},
    {"primitive"},
});

} // namespace
} // namespace zeno
