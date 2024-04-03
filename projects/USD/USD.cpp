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

#include <pxr/usd/usdLux/cylinderLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/geometryLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

#include <zeno/zeno.h>
#include <zeno/core/IObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/DictObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/LightObject.h>
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

zeno::vec3f _temperatureToRGB(float temperatureInKelvins) {
    // copy from ProcedrualSkeNode.cpp
    zeno::vec3f retColor;

    temperatureInKelvins = zeno::clamp(temperatureInKelvins, 1000.0f, 40000.0f) / 100.0f;

    if (temperatureInKelvins <= 66.0f)
    {
        retColor[0] = 1.0f;
        retColor[1] = zeno::clamp(0.39008157876901960784f * log(temperatureInKelvins) - 0.63184144378862745098f, 0.0f, 1.0f);
    } else {
        float t = temperatureInKelvins - 60.0f;
        retColor[0] = zeno::clamp(1.29293618606274509804f * pow(t, -0.1332047592f), 0.0f, 1.0f);
        retColor[1] = zeno::clamp(1.12989086089529411765f * pow(t, -0.0755148492f), 0.0f, 1.0f);
    }

    if (temperatureInKelvins >= 66.0f)
        retColor[2] = 1.0f;
    else if (temperatureInKelvins <= 19.0f)
        retColor[2] = 0.0f;
    else
        retColor[2] = zeno::clamp(0.54320678911019607843f * log(temperatureInKelvins - 10.0f) - 1.19625408914f, 0.0f, 1.0f);

    return retColor;
}

void _handleUSDCommonLightAttributes(const pxr::UsdPrim& usdPrim, zeno::PrimitiveObject& zPrim) {
    auto light = pxr::UsdLuxBoundableLightBase(usdPrim);

    /*** handle attributes about lighting ***/
    float intensity;
    float exposure;
    bool useTemperature;
    zeno::vec3f color;

    light.GetIntensityAttr().Get(&intensity);
    light.GetExposureAttr().Get(&exposure);
    light.GetEnableColorTemperatureAttr().Get(&useTemperature);

    if (useTemperature) {
        float temperature;
        light.GetColorTemperatureAttr().Get(&temperature);
        color = _temperatureToRGB(temperature);
    } else {
        pxr::GfVec3f _color;
        light.GetColorAttr().Get(&_color);
        color = { _color[0], _color[1], _color[2] };
    }

    auto scaler = powf(2.0f, exposure);
    if (std::isnan(scaler) || std::isinf(scaler) || scaler < 0.0f) {
        scaler = 1.0f;
    }
    color *= intensity * scaler;

    // check if color is legal
    for (float& _c : color) {
        if (std::isnan(_c) || std::isinf(_c) || _c < 0.0f) {
            _c = 1.0f;
        }
    }
    auto& verts = zPrim.verts;
    auto& colors = verts.add_attr<zeno::vec3f>("clr");
    for (auto& c : colors) {
        c = color;
    }

    // neccessary lighting info for zeno light prim
    auto typeEnum = zeno::LightType::Diffuse;
    auto typeOrder = magic_enum::enum_integer(typeEnum);

    auto shapeEnum = zeno::LightShape::TriangleMesh;
    auto shapeOrder = magic_enum::enum_integer(shapeEnum);

    zPrim.userData().set2("type", std::move(typeOrder));
    zPrim.userData().set2("shape", std::move(shapeOrder));

    zPrim.userData().set2("intensity", intensity);
    zPrim.userData().set2("color", std::move(color));

    zPrim.userData().set2("isRealTimeObject", true);
    zPrim.userData().set2("isL", true);
    zPrim.userData().set2("ivD", false);
}

// converting USD mesh to zeno mesh
void _convertMeshFromUSDToZeno(const pxr::UsdPrim& usdPrim, zeno::PrimitiveObject& zPrim) {
    const std::string& typeName = usdPrim.GetTypeName().GetString();

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
            // TODO: points, lines and error types to be considered
            ;
        }
    }
    else if (typeName == "CylinderLight") {
        auto light = pxr::UsdLuxCylinderLight(usdPrim);

        float height; // length of the cylinder in its local X axis
        float radius;
        light.GetLengthAttr().Get(&height); // so why USD calls it 'length' at CylinderLight and 'height' at Cylinder ??
        light.GetRadiusAttr().Get(&radius);

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
            verts.emplace_back(h, r0, r1);
        }
        // vertices of the bottom
        for (size_t i = 0; i < COLUMNS; i++) {
            float rad = 2.0f * M_PI * i / COLUMNS;
            float r0 = cos(rad) * radius;
            float r1 = -sin(rad) * radius;
            float h = -0.5f * height;
            verts.emplace_back(h, r0, r1);
        }
        verts.emplace_back(0.5f * height, 0.0f, 0.0f);
        verts.emplace_back(-0.5f * height, 0.0f, 0.0f);

        for (size_t i = 0; i < COLUMNS; i++) {
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

        _handleUSDCommonLightAttributes(usdPrim, zPrim);
    }
    else if (typeName == "DiskLight") {
        /*
        * Details of DiskLight from USD doc:
        * Light emitted from one side of a circular disk.
        * The disk is centered in the XY plane and emits light along the -Z axis.
        */
        auto light = pxr::UsdLuxDiskLight(usdPrim);

        float radius;
        light.GetRadiusAttr().Get(&radius);

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;

        const int COLUMNS = 32;
        verts.emplace_back(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < COLUMNS; ++i) {
            float rad = 2.0f * M_PI * i / COLUMNS;
            float x = cos(rad) * radius;
            float y = sin(rad) * radius;
            verts.emplace_back(x, y, 0.0f);
        }

        for (int i = 1; i < COLUMNS; ++i) {
            tris.emplace_back(0, i + 1, i);
        }
        // the last part
        tris.emplace_back(0, 1, COLUMNS);

        _handleUSDCommonLightAttributes(usdPrim, zPrim);
    }
    else if (typeName == "DistantLight") { // TODO: we don't support light source with no shape, so skip it for now
        /*
        * Light emitted from a distant source along the -Z axis.
        * Also known as a directional light.
        */
        /*
        auto light = pxr::UsdLuxDistantLight(usdPrim);

        _handleUSDCommonLightAttributes(usdPrim, zPrim);

        // distant light is not common at all, so we set up some variables here
        float angle;
        light.GetAngleAttr().Get(&angle);
        // TODO: angle is not used in zeno lighting for now, ignore it

        auto typeEnum = zeno::LightType::Direction;
        auto typeOrder = magic_enum::enum_integer(typeEnum);

        zPrim.userData().set2("type", std::move(typeOrder));
        */
    }
    else if (typeName == "DomeLight") {
        auto light = pxr::UsdLuxDomeLight(usdPrim);

        float radius;
        light.GetGuideRadiusAttr().Get(&radius);
        // constructing the huge sphere

        auto& verts = zPrim.verts;
        auto& tris = zPrim.tris;
        auto& uvs = zPrim.uvs; // TODO
        auto& norms = verts.add_attr<zeno::vec3f>("nrm");

        const int ROWS = 32;
        const int COLUMNS = 32;

        verts.emplace_back(0.0f, radius, 0.0f);
        norms.emplace_back(0.0f, -1.0f, 0.0f);
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
                norms.emplace_back(zeno::normalize(-zeno::vec3f(x, y, z)));
            }
        }
        verts.emplace_back(0.0f, -radius, 0.0f);
        norms.emplace_back(0.0f, 1.0f, 0.0f);

        // setup sphere poly indices
        {
            //head
            for (auto column = 0; column < COLUMNS; column++) {
                if (column == COLUMNS - 1) {
                    tris.emplace_back(0, COLUMNS, 1);
                }
                else {
                    tris.emplace_back(0, column + 1, column + 2);
                }
            }
            //body
            for (auto row = 1; row < ROWS - 1; row++) {
                for (auto column = 0; column < COLUMNS; column++) {
                    if (column == COLUMNS - 1) {
                        tris.emplace_back(
                            (row - 1) * COLUMNS + 1,
                            (row - 1) * COLUMNS + COLUMNS,
                            row * COLUMNS + COLUMNS
                        );
                        tris.emplace_back(
                            (row - 1) * COLUMNS + 1,
                            row * COLUMNS + COLUMNS,
                            row * COLUMNS + 1
                        );
                    }
                    else {
                        tris.emplace_back(
                            (row - 1) * COLUMNS + column + 2,
                            (row - 1) * COLUMNS + column + 1,
                            row * COLUMNS + column + 1
                        );
                        tris.emplace_back(
                            (row - 1)* COLUMNS + column + 2,
                            row* COLUMNS + column + 1,
                            row* COLUMNS + column + 2
                        );
                    }
                }
            }
            //tail
            for (auto column = 0; column < COLUMNS; column++) {
                if (column == COLUMNS - 1) {
                    tris.emplace_back(
                        (ROWS - 2)* COLUMNS + 1,
                        (ROWS - 2)* COLUMNS + column + 1,
                        (ROWS - 1)* COLUMNS + 1
                    );
                }
                else {
                    tris.emplace_back(
                        (ROWS - 2)* COLUMNS + column + 2,
                        (ROWS - 2)* COLUMNS + column + 1,
                        (ROWS - 1)* COLUMNS + 1
                    );
                }
            }
        }

        _handleUSDCommonLightAttributes(usdPrim, zPrim);

        // TODO: handle texture lighting
        /*
        auto texFileAttr = light.GetTextureFileAttr();
        if (texFileAttr.HasAssetInfo()) {
            pxr::SdfAssetPath texPath;
            texFileAttr.Get(&texPath);
            const std::string& path = texPath.GetAssetPath();
            if (!path.empty()) {
                zPrim.userData().set2("lightTexture", std::move(path));
                zPrim.userData().set2("lightGamma", 1.0f); // TODO
            }
        }*/
    }
    else if (typeName == "RectLight") {
        ;
    }
    else if (typeName == "SphereLight") {
        ;
    }
    else if (typeName == "GeometryLight") {
        ;
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

// return a zeno mesh prim from the given USD mesh prim path
struct ImportUSDMesh: zeno::INode {
    virtual void apply() override {
        std::string& usdPath = get_input2<zeno::StringObject>("USDDescription")->get();
        std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();

        auto stage = pxr::UsdStage::Open(usdPath);
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

        // converting mesh
        _convertMeshFromUSDToZeno(prim, *zPrim);

        set_output2("prim", std::move(zPrim));
    }
};
ZENDEFNODE(ImportUSDMesh,
    {
        /* inputs */
        {
            {"string", "USDDescription"},
            {"string", "primPath"}
        },
        /* outputs */
        {
            {"primitive", "prim"}
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

// generate transform node from prim
struct EvalUSDPrim: zeno::INode {
    virtual void apply() override{
        ;
    }
};
ZENDEFNODE(EvalUSDPrim,
    {
        /* inputs */
        {
            {"readpath", "USDDescription"},
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
    }
);
