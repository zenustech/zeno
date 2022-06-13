#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/utils/logger.h>

#include <mutex>
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#include <xatlas.h>
#include <tiny_obj_loader.h>

class Stopwatch
{
public:
    Stopwatch() { reset(); }
    void reset() { m_start = clock(); }
    double elapsed() const { return (clock() - m_start) * 1000.0 / CLOCKS_PER_SEC; }
private:
    clock_t m_start;
};

// May be called from any thread.
static bool ProgressCallback(xatlas::ProgressCategory category, int progress, void *userData)
{
    // Don't interupt verbose printing.
    Stopwatch *stopwatch = (Stopwatch *)userData;
    static std::mutex progressMutex;
    std::unique_lock<std::mutex> lock(progressMutex);
    if (progress == 0)
        stopwatch->reset();
    printf("\r   %s [", xatlas::StringForEnum(category));
    for (int i = 0; i < 10; i++)
        printf(progress / ((i + 1) * 10) ? "*" : " ");
    printf("] %d%%", progress);
    fflush(stdout);
    if (progress == 100)
        printf("\n      %.2f seconds (%g ms) elapsed\n", stopwatch->elapsed() / 1000.0, stopwatch->elapsed());
    return true;
}

bool transform_zenoObj(std::vector<tinyobj::shape_t> shapes, xatlas::Atlas *atlas, zeno::PrimitiveObject* outprim)
{
    std::vector<zeno::vec3f> vecVerts;
    std::vector<zeno::vec3i> vecTris;
    std::vector<zeno::vec3f> vecUVs;
    std::vector<zeno::vec3f> vecNrms;

    uint32_t firstVertex = 0;
    for (uint32_t i = 0; i < atlas->meshCount; i++) {
        const xatlas::Mesh &mesh = atlas->meshes[i];
        for (uint32_t v = 0; v < mesh.vertexCount; v++) {
            const xatlas::Vertex &vertex = mesh.vertexArray[v];                
            const float *pos = &shapes[i].mesh.positions[vertex.xref * 3];
            vecVerts.push_back(zeno::vec3f(pos[0], pos[1], pos[2]));
            if (!shapes[i].mesh.normals.empty()) {
                const float *normal = &shapes[i].mesh.normals[vertex.xref * 3];
                vecNrms.push_back(zeno::vec3f(normal[0], normal[1], normal[2]));
            }    
            vecUVs.push_back(zeno::vec3f(vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height, 0));     
        }

        for (uint32_t f = 0; f < mesh.indexCount/3; f++) {
            const uint32_t index0 = firstVertex + mesh.indexArray[f*3]; // 1-indexed
            const uint32_t index1 = firstVertex + mesh.indexArray[f*3+1]; // 1-indexed
            const uint32_t index2 = firstVertex + mesh.indexArray[f*3+2]; // 1-indexed
            vecTris.push_back(zeno::vec3i(index0, index1, index2));
            // vecTris.push_back(zeno::vec3i(index1, index1, index1));
            // vecTris.push_back(zeno::vec3i(index2, index2, index2));            
        }
        firstVertex += mesh.vertexCount;
    }

    outprim->verts.resize(vecVerts.size());
    for(auto i = 0; i < vecVerts.size(); i++)
        outprim->verts[i] = vecVerts[i];

    outprim->tris.resize(vecTris.size());
    for(auto i = 0; i < vecTris.size(); i++)
        outprim->tris[i] = vecTris[i];

    auto &att_uv  = outprim->add_attr<zeno::vec3f>("uv");
    for(auto i = 0; i < vecUVs.size(); i++)
        att_uv[i] = vecUVs[i];

    if(vecNrms.size() > 0)
    {
        auto &att_nrm = outprim->add_attr<zeno::vec3f>("nrm");
        for(auto i = 0; i < vecNrms.size(); i++)
            att_nrm[i] = vecNrms[i];   
    }

    zeno::log_info("output: vertices {}", outprim->verts.size());
    zeno::log_info("output: indices {}", outprim->tris.size());

    return true;
    }

bool calcUV(std::vector<tinyobj::shape_t> shapes, zeno::PrimitiveObject* outprim)
{
    // Create empty atlas.
    xatlas::Atlas *atlas = xatlas::Create();

    // Set progress callback.
    Stopwatch globalStopwatch, stopwatch;
    xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);

    // Add meshes to atlas.
    uint32_t totalVertices = 0, totalFaces = 0;
    for (int i = 0; i < (int)shapes.size(); i++) {
        const tinyobj::mesh_t &objMesh = shapes[i].mesh;
        xatlas::MeshDecl meshDecl;
        meshDecl.vertexCount = (uint32_t)objMesh.positions.size() / 3;
        meshDecl.vertexPositionData = objMesh.positions.data();
        meshDecl.vertexPositionStride = sizeof(float) * 3;
        if (!objMesh.normals.empty()) {
            meshDecl.vertexNormalData = objMesh.normals.data();
            meshDecl.vertexNormalStride = sizeof(float) * 3;
        }
        if (!objMesh.texcoords.empty()) {
            meshDecl.vertexUvData = objMesh.texcoords.data();
            meshDecl.vertexUvStride = sizeof(float) * 2;
        }
        meshDecl.indexCount = (uint32_t)objMesh.indices.size();
        meshDecl.indexData = objMesh.indices.data();
        meshDecl.indexFormat = xatlas::IndexFormat::UInt32;

        xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, (uint32_t)shapes.size());
        if (error != xatlas::AddMeshError::Success) {
            xatlas::Destroy(atlas);
            zeno::log_error("Error adding mesh {} {}: {}", i, shapes[i].name, xatlas::StringForEnum(error));
            return false;
        }
        totalVertices += meshDecl.vertexCount;
        if (meshDecl.faceCount > 0)
            totalFaces += meshDecl.faceCount;
        else
            totalFaces += meshDecl.indexCount / 3; // Assume triangles if MeshDecl::faceCount not specified.
    }
    xatlas::AddMeshJoin(atlas); // Not necessary. Only called here so geometry totals are printed after the AddMesh progress indicator.
    zeno::log_info("total vertices: {}", totalVertices);
    zeno::log_info("total faces: {}", totalFaces);
    // Generate atlas.
    zeno::log_info("Generating atlas");

    xatlas::Generate(atlas);
    zeno::log_info("charts {}", atlas->chartCount);
    zeno::log_info("atlases {}", atlas->atlasCount);
    for (uint32_t i = 0; i < atlas->atlasCount; i++)
        zeno::log_info("{}: {}% utilization", i, atlas->utilization[i] * 100.0f);
    zeno::log_info("{}x{} resolution", atlas->width, atlas->height);
    totalVertices = 0;
    totalFaces = 0;
    for (uint32_t i = 0; i < atlas->meshCount; i++) {
        const xatlas::Mesh &mesh = atlas->meshes[i];
        totalVertices += mesh.vertexCount;
        totalFaces += mesh.indexCount/3;
        // Input and output index counts always match.
        assert(mesh.indexCount == (uint32_t)shapes[i].mesh.indices.size());
    }
    zeno::log_info("total vertices: {}", totalVertices);
    zeno::log_info("total faces: {}", totalFaces);

    auto ret = transform_zenoObj(shapes, atlas, outprim);
    if(ret == false){
        zeno::log_info("transform to zenoObj type error");
    }

    xatlas::Destroy(atlas);

    return ret;
}

bool calcUVForPath(std::string path, zeno::PrimitiveObject* outprim)
{
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    unsigned int flags = 0;

    zeno::log_info("obj Loading: {}", path);
    flags = tinyobj::triangulation; 
    if (!tinyobj::LoadObj(shapes, materials, err, path.c_str(), NULL, flags)) {
        zeno::log_error("Error: {}", err);
        return false;
    }
    if (shapes.size() == 0) {
        zeno::log_error("Error: no shapes in obj file");
        return EXIT_FAILURE;
    }
    zeno::log_info("shapes: {}", shapes.size());

    return calcUV(shapes, outprim);
}

bool calcUVForData(zeno::PrimitiveObject* inprim, zeno::PrimitiveObject* outprim)
{
    std::vector<float> v;
    for (auto p :inprim->verts)
    {
        v.push_back(p[0]);
        v.push_back(p[1]);
        v.push_back(p[2]);
    }
    std::vector<float> f;
    for (auto p :inprim->tris)
    {
        f.push_back(p[0]);
        f.push_back(p[1]);
        f.push_back(p[2]);
    }
    zeno::log_info("total vertices: {}", inprim->verts.size());
    zeno::log_info("total faces: {}", inprim->tris.size());

    std::string err;    
    tinyobj::shape_t shape;

    bool ret = TranformZenodata(shape, v, f, err);
    if (ret) {
        std::vector<tinyobj::shape_t> shapes;
        shapes.push_back(shape);
        return calcUV(shapes, outprim);
    }
    else{
        zeno::log_info("shapes: {}", err.c_str());
        return false;
    }
    return true;
}

struct CalcGeometryUV : zeno::INode{
    
    virtual void apply() override {
        auto outprim = new zeno::PrimitiveObject;

        auto path = get_input<zeno::StringObject>("objpath")->get();       

        bool ret = false;
        if(!path.empty())
        {
            ret = calcUVForPath(path, outprim);                    
        }
        else
        {
            auto prim = get_input<zeno::PrimitiveObject>("prim");
            ret = calcUVForData(prim.get(), outprim);                
        }

        if(ret == false){
            zeno::log_error("CalcGeometryUV error");
            set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(new zeno::PrimitiveObject)));        
            return;
        }
        set_output("prim", std::move(std::shared_ptr<zeno::PrimitiveObject>(outprim)));        
    }
};

ZENDEFNODE(CalcGeometryUV, 
{
    /*输入*/
    {
        {"readpath", "objpath", ""},
        {"PrimitiveObject", "prim", ""},
    },
    /*输出*/
    {   
        "prim"
    },
    /*参数*/
    {},
    /*类别*/
    {"math"}
});