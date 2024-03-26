#include <cmath>
#include <iostream>
#include <algorithm>

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/plane.h>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/extra/GlobalState.h>
#include <zeno/utils/vec.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/eulerangle.h>

#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <zeno/types/MatrixObject.h>
#include <zeno/utils/string.h>

// record usd stage path and the file pointer
struct USDDescription {
    std::string mUSDPath = "";
    pxr::UsdStageRefPtr mStage = nullptr;
};

struct USDPrimKeeper : zeno::IObject {
    pxr::UsdPrim mPrim;
};

// converting USD mesh to zeno mesh
void _convertMeshFromUSDToZeno(const pxr::UsdPrim& usdPrim, zeno::PrimitiveObject& zPrim) {
    /*
    * Converting mesh
    */
    const std::string& typeName = usdPrim.GetTypeName().GetString();

    /*
    * TODO: these codes will be refactored
    */
    if (typeName == "Mesh") {
        /*** Load from USD prim ***/
        const auto& usdMesh = pxr::UsdGeomMesh(usdPrim);
        auto verCounts = usdMesh.GetFaceVertexCountsAttr();
        auto verIndices = usdMesh.GetFaceVertexIndicesAttr();
        auto points = usdMesh.GetPointsAttr();
        auto usdNormals = usdMesh.GetNormalsAttr();
        auto extent = usdMesh.GetExtentAttr(); // bounding box
        auto vertUVs = usdPrim.GetAttribute(pxr::TfToken("primvars:st"));
        auto orient = usdMesh.GetOrientationAttr();
        auto doubleSided = usdMesh.GetDoubleSidedAttr();

        bool isDoubleSided;
        doubleSided.Get(&isDoubleSided); // TODO: double sided

        // decide whether we use left handed order to construct faces
        pxr::TfToken faceOrder;
        usdMesh.GetOrientationAttr().Get(&faceOrder);
        bool isReversedFaceOrder = (faceOrder.GetString() == "leftHanded");

        /*
        * vertexCountPerFace indicates the vertex count of each face of mesh
        * -1: not initialized
        * 0: mesh including triangles, quads or polys simutaneously, treat as poly, 0 1 2 is NOT included 
        * 1: point
        * 2: line
        * 3: triangle
        * 4: quad
        * 5 and 5+: poly
        * a mesh with 0 | 1 | 2 and 3 | 3+ will crash this code
        */
        int vertexCountPerFace = -1;
        pxr::VtArray<int> verCountValues;
        verCounts.Get(&verCountValues);
        for (const int& verCount : verCountValues) {
            if (vertexCountPerFace == -1){ // initialize face vertex count
                vertexCountPerFace = verCount;
            } else {
                if (vertexCountPerFace != verCount) {
                    // this is a poly mesh
                    vertexCountPerFace = 0;
                    break;
                }
            }
        }

        /*** Zeno Prim definition ***/
        auto& verts = zPrim.verts;

        /*** Start setting up mesh ***/
        pxr::VtArray<pxr::GfVec3f> pointValues;
        points.Get(&pointValues);
        for (const auto& point : pointValues) {
            verts.emplace_back(point.data()[0], point.data()[1], point.data()[2]);
        }

        if (vertUVs.HasValue()) {
            auto& uvs = verts.add_attr<zeno::vec2f>("uvs");
            pxr::VtArray<pxr::GfVec2f> uvValues;
            vertUVs.Get(&uvValues);
            for (const auto& uvValue : uvValues) {
                uvs.emplace_back(uvValue.data()[0], uvValue.data()[1]);
            }
        }

        if (usdNormals.HasValue()) {
            auto& norms = zPrim.verts.add_attr<zeno::vec3f>("nrm");
            pxr::VtArray<pxr::GfVec3f> normalValues;
            usdNormals.Get(&normalValues);
            for (const auto& normalValue : normalValues) {
                norms.emplace_back(normalValue.data()[0], normalValue.data()[1], normalValue.data()[2]);
            }
        }

        pxr::VtArray<int> indexValues;
        verIndices.Get(&indexValues);

        if (vertexCountPerFace == 3) { // triangle mesh
            auto& tris = zPrim.tris;
            for (int start = 0; start < indexValues.size(); start += vertexCountPerFace) {
                if (isReversedFaceOrder) {
                    tris.emplace_back(
                        indexValues[start],
                        indexValues[start + 2],
                        indexValues[start + 1]
                    );
                } else {
                    tris.emplace_back(
                        indexValues[start],
                        indexValues[start + 1],
                        indexValues[start + 2]
                    );
                }
            }
        } else if (vertexCountPerFace == 4) { // quad mesh
            auto& quads = zPrim.quads;
            for (int start = 0; start < indexValues.size(); start += vertexCountPerFace) {
                if (isReversedFaceOrder) {
                    quads.emplace_back(
                        indexValues[start + 3],
                        indexValues[start + 2],
                        indexValues[start + 1],
                        indexValues[start]
                    );
                } else {
                    quads.emplace_back(
                        indexValues[start],
                        indexValues[start + 1],
                        indexValues[start + 2],
                        indexValues[start + 3]
                    );
                }
            }
        } else if (vertexCountPerFace >= 5 || vertexCountPerFace == 0) { // poly mesh
            auto& polys = zPrim.polys;
            auto& loops = zPrim.loops;
            int start = 0;
            for (int verFaceCount : verCountValues) {
                for (int subFaceIndex = 0; subFaceIndex < verFaceCount; ++subFaceIndex) {
                    if (isReversedFaceOrder) {
                        loops.emplace_back(indexValues[start + verFaceCount - 1 - subFaceIndex]);
                    } else {
                        loops.emplace_back(indexValues[start + subFaceIndex]);
                    }
                }
                polys.emplace_back(start, verFaceCount);
                start += verFaceCount;
            }
        } else {
            // TODO: points, lines and errors to be considered
            ;
        }
    }
    else if (typeName == "Sphere") {
        auto sphere = pxr::UsdGeomSphere(usdPrim);
        pxr::VtValue radiusValue;
        sphere.GetRadiusAttr().Get(&radiusValue);
        float radius = static_cast<float>(radiusValue.Get<double>());

        auto& verts = zPrim.verts;
        auto& polys = zPrim.polys;
        auto& loops = zPrim.loops;
        auto& uvs = zPrim.uvs;
        auto& norms = verts.add_attr<zeno::vec3f>("nrm");

        const int ROWS = 30;
        const int COLUMNS = 30;

        verts.emplace_back(0.0f, radius, 0.0f);
        for (int row = 1; row < ROWS; row++) {
            float v = 1.0f * row / ROWS;
            float theta = M_PI * v;
            for (int column = 0; column < COLUMNS; column++) {
                float u = 1.0f * column / COLUMNS;
                float phi = M_PI * 2 * u;
                float x = radius * sin(theta) * cos(phi);
                float y = radius * cos(theta);
                float z = radius * -sin(theta) * sin(phi);
                verts.emplace_back(x, y, z);
            }
        }
        verts.emplace_back(0.0f, -radius, 0.0f);

        // setup sphere poly indices
        {
            //head
            for (auto column = 0; column < COLUMNS; column++) {
                if (column == COLUMNS - 1) {
                    loops.emplace_back(0);
                    loops.emplace_back(COLUMNS);
                    loops.emplace_back(1);
                    polys.emplace_back(column * 3, 3);
                } else {
                    loops.emplace_back(0);
                    loops.emplace_back(column + 1);
                    loops.emplace_back(column + 2);
                    polys.emplace_back(column * 3, 3);
                }
            }
            //body
            for (auto row = 1; row < ROWS - 1; row++) {
                for (auto column = 0; column < COLUMNS; column++) {
                    if (column == COLUMNS - 1) {
                        loops.emplace_back((row - 1) * COLUMNS + 1);
                        loops.emplace_back((row - 1) * COLUMNS + COLUMNS);
                        loops.emplace_back(row * COLUMNS + COLUMNS);
                        loops.emplace_back(row * COLUMNS + 1);
                        polys.emplace_back(COLUMNS * 3 + (row - 1) * COLUMNS * 4 + column * 4, 4);
                    } else {
                        loops.emplace_back((row - 1) * COLUMNS + column + 2);
                        loops.emplace_back((row - 1) * COLUMNS + column + 1);
                        loops.emplace_back(row * COLUMNS + column + 1);
                        loops.emplace_back(row * COLUMNS + column + 2);
                        polys.emplace_back(loops.size() - 4, 4);
                    }
                }
            }
            //tail
            for (auto column = 0; column < COLUMNS; column++) {
                if (column == COLUMNS - 1) {
                    loops.emplace_back((ROWS - 2) * COLUMNS + 1);
                    loops.emplace_back((ROWS - 2) * COLUMNS + column + 1);
                    loops.emplace_back((ROWS - 1) * COLUMNS + 1);
                    polys.emplace_back(COLUMNS * 3 + (ROWS - 2) * COLUMNS * 4 + column * 3, 3);
                } else {
                    loops.emplace_back((ROWS - 2) * COLUMNS + column + 2);
                    loops.emplace_back((ROWS - 2) * COLUMNS + column + 1);
                    loops.emplace_back((ROWS - 1) * COLUMNS + 1);
                    polys.emplace_back(loops.size() - 3, 3);
                }
            }
        }

    }
    else if (typeName == "Cube") {
        auto sizeAttr = usdPrim.GetAttribute(pxr::TfToken("size"));
        double size, halfSize;
        sizeAttr.Get(&size);
        halfSize = size * 0.5;

        // use quad mode as default
        auto& verts = zPrim.verts;
        auto& polys = zPrim.polys;
        auto& loops = zPrim.loops;

        // TODO: support uv
        // auto& uvs = zPrim.uvs;

        for (int y = -1; y <= 1; y += 2) {
            for (int z = -1; z <= 1; z += 2) {
                for (int x = -1; x <= 1; x += 2) {
                    verts.emplace_back(x * halfSize, y * halfSize, z * halfSize);
                }
            }
        }

        static const int CUBE_FACE_INDICES[] = {
                4, 6, 7, 5, // up
                0, 1, 3, 2, // down
                0, 2, 6, 4, // left
                1, 5, 7, 3, // right
                0, 4, 5, 1, // front
                2, 3, 7, 6 // back
        };
        int faceStartIndex = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 4; ++j) {
                loops.emplace_back(CUBE_FACE_INDICES[i * 4 + j]);
            }
            polys.emplace_back(faceStartIndex, 4);
            faceStartIndex += 4;
        }
        // TODO: normals ?
    }
    else if (typeName == "Cylinder") {
        auto cylinder = pxr::UsdGeomCylinder(usdPrim);
        
        /*** Read from USD ***/
        char axis;
        float height;
        float radius;
        pxr::VtValue heightValue;
        pxr::VtValue radiusValue;
        pxr::VtValue axisValue;
        cylinder.GetHeightAttr().Get(&heightValue);
        cylinder.GetRadiusAttr().Get(&radiusValue);
        cylinder.GetAxisAttr().Get(&axisValue);
        height = static_cast<float>(heightValue.Get<double>());
        radius = static_cast<float>(radiusValue.Get<double>());
        axis = axisValue.Get<pxr::TfToken>().GetString()[0];

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;

        /*** Constructing cylinder ***/
        const int COLUMNS = 32;
        // vertices of the top
        for (size_t i = 0; i < COLUMNS; i++) {
            float rad = 2.0f * M_PI * i / COLUMNS;
            float r0 = cos(rad) * radius;
            float r1 = -sin(rad) * radius;
            float h = 0.5f * height;
            if (axis == 'Z') {
                std::swap(r1, h);
            } else if (axis == 'X') {
                std::swap(r0, h);
            } else {
                // we use Y-axis as default, no need to exchange
            }
            verts.emplace_back(r0, h, r1);
        }
        // vertices of the bottom
        for (size_t i = 0; i < COLUMNS; i++) {
            float rad = 2.0f * M_PI * i / COLUMNS;
            float r0 = cos(rad) * radius;
            float r1 = -sin(rad) * radius;
            float h = -0.5f * height;
            if (axis == 'Z') {
                std::swap(r1, h);
            } else if (axis == 'X') {
                std::swap(r0, h);
            } else {
                // we use Y-axis as default, no need to exchange
            }
            verts.emplace_back(r0, h, r1);
        }
        if (axis == 'Z') {
            verts.emplace_back(0.0f, 0.0f, 0.5f * height);
            verts.emplace_back(0.0f, 0.0f, -0.5f * height);
        } else if (axis == 'X') {
            verts.emplace_back(0.5f * height, 0.0f, 0.0f);
            verts.emplace_back(-0.5f * height, 0.0f, 0.0f);
        } else {
            verts.emplace_back(0.0f, 0.5f * height, 0.0f);
            verts.emplace_back(0.0f, -0.5f * height, 0.0f);
        }

        for (size_t i = 0; i < COLUMNS; i++) {
            if (axis != 'Y') {
                tris.emplace_back(COLUMNS * 2, (i + 1) % COLUMNS, i); // top
                tris.emplace_back(i + COLUMNS, (i + 1) % COLUMNS + COLUMNS, COLUMNS * 2 + 1); // bottom

                // side
                size_t _0 = i;
                size_t _1 = (i + 1) % COLUMNS;
                size_t _2 = (i + 1) % COLUMNS + COLUMNS;
                size_t _3 = i + COLUMNS;
                tris.emplace_back(_1, _2, _0);
                tris.emplace_back(_2, _3, _0);
            }
            else {
                tris.emplace_back(COLUMNS * 2, i, (i + 1) % COLUMNS); // top
                tris.emplace_back(i + COLUMNS, COLUMNS * 2 + 1, (i + 1) % COLUMNS + COLUMNS); // bottom

                // side
                size_t _0 = i;
                size_t _1 = (i + 1) % COLUMNS;
                size_t _2 = (i + 1) % COLUMNS + COLUMNS;
                size_t _3 = i + COLUMNS;
                tris.emplace_back(_1, _0, _2);
                tris.emplace_back(_2, _0, _3);
            }
        }
    }
    else if (typeName == "Cone") {
        auto cone = pxr::UsdGeomCone(usdPrim);

        /*** Read from USD ***/
        char axis;
        float height;
        float radius;
        pxr::VtValue heightValue;
        pxr::VtValue radiusValue;
        pxr::VtValue axisValue;
        cone.GetHeightAttr().Get(&heightValue);
        cone.GetRadiusAttr().Get(&radiusValue);
        cone.GetAxisAttr().Get(&axisValue);
        height = static_cast<float>(heightValue.Get<double>());
        radius = static_cast<float>(radiusValue.Get<double>());
        axis = axisValue.Get<pxr::TfToken>().GetString()[0];

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;
        /*** Constructing cylinder ***/
        const int COLUMNS = 32;
        // vertices of the bottom
        for (size_t i = 0; i < COLUMNS; i++) {
            float rad = 2.0f * M_PI * i / COLUMNS;
            float r0 = cos(rad) * radius;
            float r1 = -sin(rad) * radius;
            float h = -0.5f * height;
            if (axis == 'Z') {
                std::swap(r1, h);
            } else if (axis == 'X') {
                std::swap(r0, h);
            } else {
                // we use Y-axis as default, no need to exchange
            }
            verts.emplace_back(r0, h, r1);
        }
        if (axis == 'Z') {
            verts.emplace_back(0.0f, 0.0f, 0.5f * height);
            verts.emplace_back(0.0f, 0.0f, -0.5f * height);
        }
        else if (axis == 'X') {
            verts.emplace_back(0.5f * height, 0.0f, 0.0f);
            verts.emplace_back(-0.5f * height, 0.0f, 0.0f);
        }
        else {
            verts.emplace_back(0.0f, 0.5f * height, 0.0f);
            verts.emplace_back(0.0f, -0.5f * height, 0.0f);
        }

        for (size_t i = 0; i < COLUMNS; i++) {
            if (axis != 'Y') {
                tris.emplace_back(COLUMNS, (i + 1) % COLUMNS, i);
                tris.emplace_back(i, (i + 1) % COLUMNS, COLUMNS + 1);
            }
            else {
                tris.emplace_back(COLUMNS, i, (i + 1) % COLUMNS);
                tris.emplace_back(i, COLUMNS + 1, (i + 1) % COLUMNS);
            }
        }
    }
    else if (typeName == "Capsule") {
        auto capsule = pxr::UsdGeomCapsule(usdPrim);

        /*** Read from USD ***/
        char axis;
        float height;
        float radius;
        pxr::VtValue heightValue;
        pxr::VtValue radiusValue;
        pxr::VtValue axisValue;
        capsule.GetHeightAttr().Get(&heightValue);
        capsule.GetRadiusAttr().Get(&radiusValue);
        capsule.GetAxisAttr().Get(&axisValue);
        height = static_cast<float>(heightValue.Get<double>());
        radius = static_cast<float>(radiusValue.Get<double>());
        axis = axisValue.Get<pxr::TfToken>().GetString()[0];

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;
        const int COLUMNS = 32;
        ; // TODO

    }
    else if (typeName == "Plane") {
        auto plane = pxr::UsdGeomPlane(usdPrim);

        char axis;
        bool doubleSided; // TODO
        float length;
        float width;
        pxr::VtValue tempValue;

        plane.GetDoubleSidedAttr().Get(&doubleSided);

        plane.GetAxisAttr().Get(&tempValue);
        axis = tempValue.Get<pxr::TfToken>().GetString()[0];

        plane.GetLengthAttr().Get(&tempValue);
        length = static_cast<float>(tempValue.Get<double>());

        plane.GetWidthAttr().Get(&tempValue);
        width = static_cast<float>(tempValue.Get<double>());

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;
        auto& uvs = verts.add_attr<zeno::vec3f>("uv");
        auto& normals = verts.add_attr<zeno::vec3f>("nrm");

        if (axis == 'Z') {
            verts.emplace_back(-0.5f * width, 0.5f * length, 0.0f);
            verts.emplace_back(0.5f * width, 0.5f * length, 0.0f);
            verts.emplace_back(-0.5f * width, -0.5f * length, 0.0f);
            verts.emplace_back(0.5f * width, -0.5f * length, 0.0f);
            for (int i = 0; i < 4; ++i) {
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
        } else if (axis == 'X') {
            verts.emplace_back(0.0f, 0.5f * length, 0.5f * width);
            verts.emplace_back(0.0f, 0.5f * length, -0.5f * width);
            verts.emplace_back(0.0f, -0.5f * length, 0.5f * width);
            verts.emplace_back(0.0f, -0.5f * length, -0.5f * width);
            for (int i = 0; i < 4; ++i) {
                normals.emplace_back(1.0f, 0.0f, 0.0f);
            }
        } else {
            verts.emplace_back(-0.5f * width, 0.0f, -0.5f * length);
            verts.emplace_back(0.5f * width, 0.0f, -0.5f * length);
            verts.emplace_back(-0.5f * width, 0.0f, 0.5f * length);
            verts.emplace_back(0.5f * width, 0.0f, 0.5f * length);
            for (int i = 0; i < 4; ++i) {
                normals.emplace_back(0.0f, 1.0f, 0.0f);
            }
        }

        // uvs
        uvs.emplace_back(0.0f, 0.0f, 0.0f);
        uvs.emplace_back(1.0f, 0.0f, 0.0f);
        uvs.emplace_back(0.0f, 1.0f, 0.0f);
        uvs.emplace_back(1.0f, 1.0f, 0.0f);

        // indices
        tris.emplace_back(0, 2, 1);
        tris.emplace_back(1, 2, 3);
    }
    else {
        // other geometry types are not supported yet
        std::cout << "Found not-supported geom type: " << typeName << std::endl;
    }
}

zeno::MatrixObject _getTransformMartrixFromUSDPrim(const pxr::UsdPrim& usdPrim) {
    /*
    * extract matrices from usd matrices
    */
    bool resetsXformStack;
    glm::mat4 finalMat(1.0f);
    glm::mat4 tempMat;
    auto xformOps = pxr::UsdGeomXform(usdPrim).GetOrderedXformOps(&resetsXformStack);
    if (xformOps.size() > 0) {
        for (auto& xformOp : xformOps) {
            pxr::GfMatrix4d transMatrix;
            transMatrix = xformOp.GetOpTransform(pxr::UsdTimeCode::Default());
            double* matValues = transMatrix.data();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    tempMat[i][j] = static_cast<float>(matValues[i * 4 + j]);
                }
            }
            finalMat = tempMat * finalMat;
        }
    }

    auto ret = zeno::MatrixObject();
    ret.m = finalMat;
    return ret;
}

/*
* Manager USDStage handles
*/
class USDDescriptionManager {
public:
    static USDDescriptionManager& instance() {
        if (!_instance) {
            _instance = new USDDescriptionManager;
        }
        return *_instance;
    }

    USDDescription& getOrCreateDescription(const std::string& usdPath) {
        auto it = mStageMap.find(usdPath);
        if (it != mStageMap.end()) {
            return it->second;
        }
        auto& stageNode = mStageMap[usdPath];
        stageNode.mUSDPath = usdPath;
        stageNode.mStage = pxr::UsdStage::Open(usdPath);
        return stageNode;
    }

    // TODO: onDestroy ?
private:
    static USDDescriptionManager* _instance;

    static USDDescription ILLEGAL_DESC;

    // store the relationship between .usd and prims
    std::map<std::string, USDDescription> mStageMap;
};

USDDescription USDDescriptionManager::ILLEGAL_DESC = USDDescription();
USDDescriptionManager* USDDescriptionManager::_instance = nullptr;

struct ReadUSD : zeno::INode {
    virtual void apply() override {
        const auto& usdPath = get_input2<zeno::StringObject>("path")->get();

        USDDescriptionManager::instance().getOrCreateDescription(usdPath);

        set_output2("USDDescription", usdPath);
    }
};
ZENDEFNODE(ReadUSD,
    {
        /* inputs */
        {
            {"readpath", "path"}
        },
        /* outputs */
        {
            {"string", "USDDescription"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);

// return a zeno prim from the given use prim path
struct ImportUSDPrim : zeno::INode {
    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();
        std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();

        auto& stageDesc = USDDescriptionManager::instance().getOrCreateDescription(usdPath);
        auto stage = stageDesc.mStage;
        if (stage == nullptr) {
            std::cout << "failed to find usd description for " << usdPath;
            return;
        }

        auto prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
        if (!prim.IsValid()) {
            std::cout << "[ImportUSDPrim] failed to import prim at " << primPath << std::endl;
            return;
        }

        auto zPrim = std::make_shared<zeno::PrimitiveObject>();
        zeno::UserData& primData = zPrim->userData();

        // construct geometry from USD format
        _convertMeshFromUSDToZeno(prim, *zPrim);

        auto mat = std::make_shared<zeno::MatrixObject>(_getTransformMartrixFromUSDPrim(prim));

        primData.set2("_usdStagePath", usdPath);
        primData.set2("_usdPrimPath", primPath);
        primData.set2("_usdPrimType", prim.GetTypeName().GetString());

        set_output2("prim", std::move(zPrim));
        set_output("xformMatrix", std::move(mat));
        zeno::zany any;
    }
};
ZENDEFNODE(ImportUSDPrim,
    {
        /* inputs */
        {
            {"string", "USDDescription"},
            {"string", "primPath"}
        },
        /* outputs */
        {
            {"primitive", "prim"},
            {"Matrix", "xformMatrix"},
            //{"skeleton"}, // USDPrimKeeper
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);

struct ViewUSDTree : zeno::INode {
    int _getDepth(const std::string& primPath) const {
        int depth = 0;
        for (char ch : primPath) {
            if (ch == '/') {
                ++depth;
            }
        }
        return depth;
    }

    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();
        auto stage = USDDescriptionManager::instance().getOrCreateDescription(usdPath).mStage;
        if (stage == nullptr) {
            std::cerr << "failed to find usd description for " << usdPath << std::endl;
            return;
        }

        auto range = stage->Traverse();

        for (auto prim : range) {
            const std::string& primPath = prim.GetPath().GetString();
            int depth = _getDepth(primPath) - 1;
            for (int i = 0; i < depth; ++i) {
                std::cout << '\t';
            }
            std::cout << '[' << prim.GetTypeName() << "] " << prim.GetName() << std::endl;
        }
    }
};
ZENDEFNODE(ViewUSDTree,
    {
        /* inputs */
        {
            {"string", "USDDescription"}
        },
    /* outputs */
    {
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

/*
* Show all prims' info of the given USD, including their types, paths and properties.
*/
struct USDShowAllPrims : zeno::INode {
    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();

        auto& usdManager = USDDescriptionManager::instance();
        auto stage = usdManager.getOrCreateDescription(usdPath).mStage;
        if (stage== nullptr) {
            std::cerr << "failed to find usd description for " << usdPath << std::endl;
            return;
        }

        // traverse and get description of all prims
        auto range = stage->Traverse();
        for (auto it : range) {
            // handle USD scene, traverse and construct zeno graph
            const std::string& primType = it.GetTypeName().GetString();
            const std::string& primPath = it.GetPath().GetString();

            std::cout << "[TYPE] " << primType << " [PATH] " << primPath << std::endl;
            const auto& attributes = it.GetAttributes();
            const auto& relations = it.GetRelationships();
            std::cout << "[Relationships] ";
            for (const auto& relation : relations) {
                pxr::SdfPathVector targets;
                relation.GetTargets(&targets);
                if (targets.empty()) {
                    continue;
                }

                std::cout << relation.GetName().GetString() << '\t';
            }
            std::cout << std::endl << "[Attributes] ";
            for (const auto& attr : attributes) {
                if (!attr.IsValid() || !attr.HasValue()) {
                    continue;
                }
                std::cout << "[" << attr.GetTypeName().GetType().GetTypeName() << "]" << attr.GetName().GetString() << '\t';
            }
            std::cout << '\n' << std::endl;
        }
    }
};
ZENDEFNODE(USDShowAllPrims,
    {
        /* inputs */
        {
            {"string", "USDDescription"}
        },
    /* outputs */
    {
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

/*
* Show userData of the given prim, in key-value format
*/
struct ShowPrimUserData : zeno::INode {
    virtual void apply() override {
        auto prim = get_input2<zeno::PrimitiveObject>("prim");
        auto& userData = prim->userData();

        std::cout << "showing userData for prim:" << std::endl;
        for (const auto& data : userData) {
            std::cout << "[Key] " << data.first << " [Value] " << data.second->as<zeno::StringObject>()->get() << std::endl;
        }
    }
};
ZENDEFNODE(ShowPrimUserData,
    {
    /* inputs */
    {
        {"primitive", "prim"},
    },
    /* outputs */
    {
        // {"primitive", "prim"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

/*
* Show all attributes and their values of a USD prim, for dev
*/
struct ShowUSDPrimAttribute : zeno::INode {
    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();
        std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();
        std::string& attrName = get_input2<zeno::StringObject>("attributeName")->get();

        auto& stageDesc = USDDescriptionManager::instance().getOrCreateDescription(usdPath);
        auto stage = stageDesc.mStage;
        if (stage == nullptr) {
            std::cerr << "failed to find usd description for " << usdPath;
            return;
        }

        auto prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
        if (!prim.IsValid()) {
            std::cout << "[ShowUSDPrimAttribute] failed to find prim at " << primPath << std::endl;
            return;
        }

        std::cout << "Showing attributes for prim: " << primPath << std::endl;
        if (attrName.empty()) { // showing all prims in the stage
            auto attributes = prim.GetAttributes();
            for (auto& attr : attributes) {
                if (!attr.IsValid() || !attr.HasValue()) {
                    continue;
                }
                pxr::VtValue val;
                attr.Get(&val);
                if (val.IsArrayValued() && val.GetArraySize() == 0) {
                    continue;
                }

                std::cout << "[Attribute Name] " << attr.GetName().GetString() << " [Attribute Type] " << attr.GetTypeName().GetCPPTypeName();
                if (val.IsArrayValued()) {
                    std::cout << " [Array Size] " << val.GetArraySize();
                }
                std::cout << "\n[Attribute Value] " << val << '\n' << std::endl;
            }
        }
        else { // showing indicated prim
            auto attr = prim.GetAttribute(pxr::TfToken(attrName));
            pxr::VtValue val;
            attr.Get(&val);
            std::cout << "[Attribute Name] " << attr.GetName().GetString() << " [Attribute Type] " << attr.GetTypeName().GetCPPTypeName();
            if (val.IsArrayValued()) {
                std::cout << " [Array Size] " << val.GetArraySize();
            }
            std::cout << "\n[Attribute Value] " << val << '\n' << std::endl;
        }
    }
};
ZENDEFNODE(ShowUSDPrimAttribute,
    {
        /* inputs */
        {
            {"string", "USDDescription"},
            {"string", "primPath"},
            {"string", "attributeName", ""}
        },
        /* outputs */
        {
            // {"primitive", "prim"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    });

struct ShowUSDPrimRelationShip : zeno::INode {
    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();
        std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();

        auto& stageDesc = USDDescriptionManager::instance().getOrCreateDescription(usdPath);
        auto stage = stageDesc.mStage;
        if (stage == nullptr) {
            std::cerr << "failed to find usd description for " << usdPath;
            return;
        }

        auto prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
        if (!prim.IsValid()) {
            std::cout << "[ShowUSDPrimAttribute] failed to find prim at " << primPath << std::endl;
            return;
        }

        std::cout << "Showing relationships for prim: " << primPath << std::endl;

        auto relations = prim.GetRelationships();
        for (auto& relation : relations) {
            pxr::SdfPathVector targets;
            relation.GetTargets(&targets);
            if (targets.size() == 0) {
                continue;
            }
            std::cout << "[Relation Name] " << relation.GetName() << std::endl;
            for (auto& target : targets) {
                std::cout << "[Relation Target] " << target.GetAsString() << std::endl;
            }
            std::cout << std::endl;
        }
    }
};
ZENDEFNODE(ShowUSDPrimRelationShip,
    {
        /* inputs */
        {
            {"string", "USDDescription"},
            {"string", "primPath"}
        },
    /* outputs */
    {
        // {"primitive", "prim"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

// convert USD prim to zeno prim
struct USDToZenoPrim : zeno::INode {
    virtual void apply() override {
        ;
    }
};
ZENDEFNODE(USDToZenoPrim,
    {
        /* inputs */
        {
            {"string", "USDDescription"},
            {"string", "primPath"},
            {"int", "frame"}
        },
    /* outputs */
    {
        // {"primitive", "prim"}
    },
    /* params */
    {},
    /* category */
    {"USD"}
    });

// generate transform node from prim
struct EvalUSDXform : zeno::INode {
    virtual void apply() override{
        ;
    }
};
ZENDEFNODE(EvalUSDXform,
    {
        /* inputs */
        {
            {"primitive", "prim"}
        },
        /* outputs */
        {
            // {"primitive", "prim"}
        },
        /* params */
        {},
        /* category */
        {"USD"}
    }
);

struct USDOpinion : zeno::INode {
    ;
};


struct USDSublayer : zeno::INode {
    virtual void apply() override {
        ;
    }
};

struct USDCollapse : zeno::INode {
    virtual void apply() override {
        ;
    }
};

struct USDSave : zeno::INode {
    virtual void apply() override {
        ;
    }
};
