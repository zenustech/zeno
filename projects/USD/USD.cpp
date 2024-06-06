#include <cmath>
#include <iostream>
#include <algorithm>

#include <pxr/pxr.h>
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/tf/type.h"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>

#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/plane.h>

#include "pxr/usd/usdSkel/skeleton.h"
#include "pxr/usd/usdSkel/animation.h"
#include "pxr/usd/usdSkel/animQuery.h"
#include "pxr/usd/usdSkel/skeletonQuery.h"
#include "pxr/usd/usdSkel/skinningQuery.h"
#include "pxr/usd/usdSkel/bindingAPI.h"
#include "pxr/usd/usdSkel/cache.h"
#include "pxr/usd/usdSkel/root.h"
#include "pxr/usd/usdSkel/utils.h"
#include "pxr/usd/usdSkel/blendShapeQuery.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>

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
#include <USD.h>

void _applyBlendShape(pxr::VtArray<pxr::GfVec3f>& points, pxr::UsdSkelBindingAPI& skelBinding, pxr::UsdSkelSkeletonQuery& skelQuery, pxr::UsdSkelSkinningQuery& skinQuery, float time) {
    if (!skinQuery.HasBlendShapes()) {
        return;
    }

    const pxr::UsdSkelAnimQuery& animQuery = skelQuery.GetAnimQuery();
    if (!animQuery) {
        return;
    }

    pxr::VtArray<float> weights;
    pxr::VtArray<float> realWeights;
    if (!animQuery.ComputeBlendShapeWeights(&weights, time)) {
        return;
    }

    if (skinQuery.GetBlendShapeMapper()) {
        if (!skinQuery.GetBlendShapeMapper()->Remap(weights, &realWeights)) {
            return;
        }
    } else {
        realWeights = std::move(weights);
    }

    pxr::UsdSkelBlendShapeQuery blendShapeQuery(skelBinding);
    pxr::VtArray<float> subShapeWeights;
    pxr::VtArray<unsigned int> blendShapeIndices;
    pxr::VtArray<unsigned int> subShapeIndices;
    if (!blendShapeQuery.ComputeSubShapeWeights(realWeights, &subShapeWeights, &blendShapeIndices, &subShapeIndices)) {
        return;
    }

    blendShapeQuery.ComputeDeformedPoints(
        pxr::TfMakeSpan(subShapeWeights),
        pxr::TfMakeSpan(blendShapeIndices),
        pxr::TfMakeSpan(subShapeIndices),
        blendShapeQuery.ComputeBlendShapePointIndices(),
        blendShapeQuery.ComputeSubShapePointOffsets(),
        pxr::TfMakeSpan(points)
    );
}

void _applySkinning(pxr::VtArray<pxr::GfVec3f>& points, pxr::VtArray<pxr::GfVec3f>& normals /* TODO */, const pxr::UsdGeomMesh& mesh, float time) {
    pxr::UsdSkelCache skelCache;
    pxr::UsdSkelBindingAPI skelBinding(mesh.GetPrim());

    auto skelRoot = pxr::UsdSkelRoot::Find(mesh.GetPrim());
    if (!skelRoot) {
        // no skelRoot found for this mesh
        return;
    }

    skelCache.Populate(skelRoot, pxr::UsdTraverseInstanceProxies());
    pxr::UsdSkelSkeletonQuery skelQuery = skelCache.GetSkelQuery(skelBinding.GetInheritedSkeleton());
    pxr::UsdSkelSkinningQuery skinQuery = skelCache.GetSkinningQuery(mesh.GetPrim());

    pxr::VtArray<pxr::GfMatrix4d> skinningTransforms;
    if (!skelQuery.ComputeSkinningTransforms(&skinningTransforms, time)) {
        return;
    }

    _applyBlendShape(points, skelBinding, skelQuery, skinQuery, time);

    if (!skinQuery.ComputeSkinnedPoints(skinningTransforms, &points, time)) {
        return;
    }

    pxr::GfMatrix4d bindTransform = skinQuery.GetGeomBindTransform(time).GetInverse();
    for (auto& point : points) {
        point = bindTransform.Transform(point);
    }

    // TODO: normals
}

// converting USD mesh to zeno mesh
void _convertMeshFromUSDToZeno(const pxr::UsdPrim& usdPrim, zeno::PrimitiveObject& zPrim, float time) {
    const std::string& typeName = usdPrim.GetTypeName().GetString();

    if (typeName != "Mesh") {
        return;
    }

    /*** Read from USD prim ***/
    const auto& usdMesh = pxr::UsdGeomMesh(usdPrim);
    auto extent = usdMesh.GetExtentAttr(); // bounding box

    bool isDoubleSided;
    usdMesh.GetDoubleSidedAttr().Get(&isDoubleSided); // TODO: double sided

    // decide whether we use left handed order to construct faces
    pxr::TfToken faceOrder;
    usdMesh.GetOrientationAttr().Get(&faceOrder, time);
    bool isReversedFaceOrder = (faceOrder.GetString() == "leftHanded");

    /*** Zeno Prim definition ***/
    auto& verts = zPrim.verts;

    /*** Start setting up mesh ***/
    pxr::VtArray<pxr::GfVec3f> pointValues;
    usdMesh.GetPointsAttr().Get(&pointValues, time);
    _applySkinning(pointValues, pointValues, usdMesh, time);
    for (const auto& point : pointValues) {
        verts.emplace_back(point.data()[0], point.data()[1], point.data()[2]);
    }

    auto vertColor = usdMesh.GetDisplayColorAttr();
    if (vertColor.HasValue()) {
        pxr::VtArray<pxr::GfVec3f> _cc;
        vertColor.Get(&_cc);
        if (_cc.size() > 0) {
            zeno::vec3f meshColor = { _cc[0][0], _cc[0][1], _cc[0][2] };
            auto& vColors = verts.add_attr<zeno::vec3f>("clr");
            for (auto& vColor: vColors) {
                vColor = meshColor;
            }
        }
    }

    auto usdNormals = usdMesh.GetNormalsAttr();
    if (usdNormals.HasValue()) {
        auto& norms = zPrim.verts.add_attr<zeno::vec3f>("nrm");
        pxr::VtArray<pxr::GfVec3f> normalValues;
        usdNormals.Get(&normalValues);
        for (const auto& normalValue : normalValues) {
            norms.emplace_back(normalValue.data()[0], normalValue.data()[1], normalValue.data()[2]);
        }
    }

    // constructing faces, treat all meshes as poly
    pxr::VtArray<int> faceSizeValues; // numbers of vertices for each face
    pxr::VtArray<int> usdIndices;
    usdMesh.GetFaceVertexCountsAttr().Get(&faceSizeValues);
    usdMesh.GetFaceVertexIndicesAttr().Get(&usdIndices);
    auto& polys = zPrim.polys;
    auto& loops = zPrim.loops;
    int start = 0;
    for (int faceSize : faceSizeValues) {
        for (int subFaceIndex = 0; subFaceIndex < faceSize; ++subFaceIndex) {
            if (isReversedFaceOrder) {
                loops.emplace_back(usdIndices[start + faceSize - 1 - subFaceIndex]);
            } else {
                loops.emplace_back(usdIndices[start + subFaceIndex]);
            }
        }
        polys.emplace_back(start, faceSize);
        start += faceSize;
    }

    auto vertUVs = usdPrim.GetAttribute(pxr::TfToken("primvars:st"));
    if (vertUVs.HasValue()) {
        pxr::VtArray<pxr::GfVec2f> usdUVs;
        vertUVs.Get(&usdUVs);

        auto& uvs = zPrim.uvs;
        uvs.resize(usdUVs.size());
        for (int i = 0; i < usdUVs.size(); ++i) {
            uvs[i] = {usdUVs[i][0], usdUVs[i][1]};
        }

        auto uvIndicesAttr = usdPrim.GetAttribute(pxr::TfToken("primvars:st:indices"));
        if (uvIndicesAttr.HasValue()) {
            pxr::VtArray<int> usdUVIndices;
            uvIndicesAttr.Get(&usdUVIndices);

            if (usdUVIndices.size() == loops.size()) { // uv index size matches vertex index size
                auto& uvIndices = zPrim.loops.add_attr<int>("uvs");
                for (int i = 0; i < loops.size(); ++i) {
                    uvIndices[i] = usdUVIndices[i];
                }
            }
            else if (usdUVIndices.size() == verts.size()) { // uv index size matches vertex size
                auto& uvIndices = zPrim.loops.add_attr<int>("uvs");
                for (int i = 0; i < loops.size(); ++i) {
                    int refVertIndex = loops[i];
                    uvIndices[i] = usdUVIndices[refVertIndex];
                }
            }
            else {
                zeno::log_error("found incorrect number of st:indices {} from prim {}", usdUVIndices.size(), usdPrim.GetPath().GetString());
            }
        }
        else {
            if (usdUVs.size() == loops.size()) {
                auto& uvIndices = zPrim.loops.add_attr<int>("uvs");
                for (int i = 0; i < loops.size(); ++i) {
                    uvIndices[i] = i;
                }
            }
            else if (usdUVs.size() == verts.size()) {
                auto& uvIndices = zPrim.loops.add_attr<int>("uvs");
                for (int i = 0; i < loops.size(); ++i) {
                    int refVertIndex = loops[i];
                    uvIndices[i] = refVertIndex;
                }
            }
            else {
                zeno::log_error("invalid st size for mesh: {} from prim {}", usdUVs.size(), usdMesh.GetPath().GetString());
            }
        }
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

// return a zeno mesh prim from the given USD mesh prim path
void ImportUSDMesh::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();
    std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();
    float frame = get_input2<float>("frame");

    auto stage = pxr::UsdStage::Open(usdPath);
    if (stage == nullptr) {
        std::cout << "failed to find usd description for " << usdPath;
        return;
    }

    if (frame < 0.0f) {
        frame = pxr::UsdTimeCode::Default().GetValue();
    }

    auto prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
    if (!prim.IsValid()) {
        std::cout << "[ImportUSDPrim] failed to import prim at " << primPath << std::endl;
        return;
    }

    auto zPrim = std::make_shared<zeno::PrimitiveObject>();
    zeno::UserData& primData = zPrim->userData();

    // converting mesh
    _convertMeshFromUSDToZeno(prim, *zPrim, frame);

    set_output2("prim", std::move(zPrim));
}

void ImportUSDPrimMatrix::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();
    std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();
    std::string& opAttrName = get_input2<zeno::StringObject>("opName")->get();
    float frameTime = get_input2<float>("frame");
    bool isInversedOp = get_input2<bool>("isInversedOp");

    auto stage = pxr::UsdStage::Open(usdPath);
    if (stage == nullptr) {
        zeno::log_error("failed to find usd description for " + usdPath);
        return;
    }

    auto prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
    if (!prim.IsValid()) {
        zeno::log_error("[ImportUSDPrim] failed to import prim at " + primPath);
        return;
    }

    if (frameTime < 0.0f) {
        frameTime = pxr::UsdTimeCode::Default().GetValue();
    }

    double* vp = nullptr;
    if (opAttrName.empty()) { // import the entire xformOp as a matrix
        auto xform = pxr::UsdGeomXform(prim);
        pxr::GfMatrix4d usdTransform = xform.ComputeLocalToWorldTransform(frameTime);
        vp = usdTransform.data();
    }
    else {
        auto attr = prim.GetAttribute(pxr::TfToken(opAttrName));
        if (!attr.IsValid()) {
            zeno::log_error("failed to get attribute named " + opAttrName);
            return;
        }

        auto op = pxr::UsdGeomXformOp(attr);
        if (op.GetOpType() == pxr::UsdGeomXformOp::TypeInvalid) {
            zeno::log_error("failed to parse xformOp cause it is an invalid type " + primPath);
            return;
        }

        pxr::VtValue mVal;
        attr.Get(&mVal, frameTime);

        vp = pxr::UsdGeomXformOp::GetOpTransform(op.GetOpType(), mVal, isInversedOp).data();
    }

    glm::mat4 realMat;
    for (int i = 0; i < 16; ++i) {
        realMat[i / 4][i % 4] = static_cast<float>(vp[i]);
    }

    auto mat = std::make_shared<zeno::MatrixObject>();
    mat->m = realMat;

    set_output2("Matrix", std::move(mat));
}

int ViewUSDTree::_getDepth(const std::string& primPath) const {
    int depth = 0;
    for (char ch : primPath) {
        if (ch == '/') {
            ++depth;
        }
    }
    return depth;
}

void ViewUSDTree::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();
    auto stage = pxr::UsdStage::Open(usdPath);
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

/*
* Show all prims' info of the given USD, including their types, paths and properties.
*/
void USDShowAllPrims::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();

    auto stage = pxr::UsdStage::Open(usdPath);
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

/*
* Show all attributes and their values of a USD prim, for dev
*/
void ShowUSDPrimAttribute::_showAttribute(std::any _attr, bool showDetail = false) const {
    auto attr = std::any_cast<const pxr::UsdAttribute&>(_attr);
    if (!showDetail && (!attr.IsValid() || !attr.HasValue())) {
        return;
    }

    pxr::VtValue val;
    attr.Get(&val);
    if (!showDetail && val.IsArrayValued() && val.GetArraySize() == 0) {
        return;
    }

    std::cout << "[Attribute Name] " << attr.GetName().GetString() << " [Attribute Type] " << attr.GetTypeName().GetCPPTypeName();
    if (val.IsArrayValued()) {
        std::cout << " [Array Size] " << val.GetArraySize();
        if (!showDetail && val.GetArraySize() > 100) {
            std::cout << "\nThis attribute value is too long to show, indicate attribute name in the node to show details.\n" << std::endl;
        }
        else {
            std::cout << "\n[Attribute Value] " << val << '\n' << std::endl;
        }
    }
    else {
        std::cout << "\n[Attribute Value] " << val << '\n' << std::endl;
    }
}

void ShowUSDPrimAttribute::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();
    std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();
    std::string& attrName = get_input2<zeno::StringObject>("attributeName")->get();

    auto stage = pxr::UsdStage::Open(usdPath);
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
            _showAttribute(attr, false);
        }
    }
    else { // showing indicated prim
        _showAttribute(prim.GetAttribute(pxr::TfToken(attrName)), true);
    }
}

void ShowUSDPrimRelationShip::apply() {
    std::string& usdPath = get_input2<zeno::StringObject>("usdPath")->get();
    std::string& primPath = get_input2<zeno::StringObject>("primPath")->get();

    auto stage = pxr::UsdStage::Open(usdPath);
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
            // continue;
        }
        std::cout << "[Relation Name] " << relation.GetName() << std::endl;
        for (auto& target : targets) {
            std::cout << "[Relation Target] " << target.GetAsString() << std::endl;
        }
        std::cout << std::endl;
    }
}
// generate transform node from prim
void EvalUSDPrim::apply() {
    ;
}
