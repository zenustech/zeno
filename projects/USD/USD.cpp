#include <iostream>

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>

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
#include <cmath>

struct USDDescription {
    std::string mUSDPath = "";
    pxr::UsdStageRefPtr mStage = nullptr;
};

// converting USD mesh to zeno mesh
void _convertPrimFromUSDToZeno(const pxr::UsdPrim& usdPrim, zeno::PrimitiveObject& zPrim) {
    /*
    * Converting mesh
    */
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
            // TODO: points, lines and errors to be considered
            ;
        }
    }
    else if (typeName == "Sphere") {
        ; // TODO
    }
    else if (typeName == "Cube") {
        ; // TODO
    }
    else {
        // not supported yet
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

    auto& ret = zeno::MatrixObject();
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
        _convertPrimFromUSDToZeno(prim, *zPrim);

        auto mat = std::make_shared<zeno::MatrixObject>(_getTransformMartrixFromUSDPrim(prim));

        primData.set2("_usdStagePath", usdPath);
        primData.set2("_usdPrimPath", primPath);
        primData.set2("_usdPrimType", prim.GetTypeName().GetString());

        set_output2("prim", std::move(zPrim));
        set_output("xformMatrix", mat);
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
            {"Matrix", "xformMatrix"}
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
        for (auto& it : range) {
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
        auto& prim = get_input2<zeno::PrimitiveObject>("prim");
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
            auto& attributes = prim.GetAttributes();
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
            auto& attr = prim.GetAttribute(pxr::TfToken(attrName));
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
