#include <zenovis/StageManager.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>

#include <pxr/usd/usdGeom/mesh.h>

#include <iostream>
#include <filesystem>

PXR_NAMESPACE_USING_DIRECTIVE

namespace zenovis {

TF_DEFINE_PRIVATE_TOKENS(
    _tokens,
    // light
    (UsdLuxDiskLight)
    (UsdLuxCylinderLight)
    (UsdLuxDistantLight)
    (UsdLuxDomeLight)
    (UsdLuxRectLight)
    (UsdLuxSphereLight)
    // prim
    (UsdGeomMesh)
    (UsdGeomCurves)
    (UsdGeomPoints)
    (UsdVolume)
    (UsdGeomCamera)
);

StageManager::StageManager(){
    zeno::log_info("USD: StageManager Constructed");

    //stagePtr = UsdStage::CreateInMemory();
    stagePtr = UsdStage::CreateNew("projects/USD/usd.usda");
};
StageManager::~StageManager(){
    zeno::log_info("USD: StageManager Destroyed");
};

int zenovis::StageManager::_UsdGeomMesh(const PrimInfo& primInfo){
    auto zenoPrim = dynamic_cast<zeno::PrimitiveObject *>(primInfo.iObject.get());
    std::filesystem::path p(primInfo.pPath); std::string nodeName = p.filename().string();
    zeno::log_info("USD: GeomMesh {}", nodeName);
    SdfPath objPath(primInfo.pPath);
    _CreateUSDHierarchy(objPath);

    UsdGeomMesh mesh = UsdGeomMesh::Define(stagePtr, objPath);
    UsdPrim usdPrim = mesh.GetPrim();

    pxr::VtArray<pxr::GfVec3f> Points;
    pxr::VtArray<pxr::GfVec3f> DisplayColor;
    pxr::VtArray<int> FaceVertexCounts;
    pxr::VtArray<int> FaceVertexIndices;

    // Points
    for(auto const& vert:zenoPrim->verts)
        Points.emplace_back(vert[0], vert[1], vert[2]);
    // Face
    if(zenoPrim->loops.size() && zenoPrim->polys.size()){
        // TODO Generate UsdGeomMesh based on these attributes
    }else{
        for(auto const& ind:zenoPrim->tris){
            FaceVertexIndices.emplace_back(ind[0]);
            FaceVertexIndices.emplace_back(ind[1]);
            FaceVertexIndices.emplace_back(ind[2]);
            FaceVertexCounts.emplace_back(3);
        }
    }
    // DisplayColor
    if(zenoPrim->verts.has_attr("clr0")){
        for(auto const& clr0:zenoPrim->verts.attr<zeno::vec3f>("clr0")){
            DisplayColor.emplace_back(clr0[0], clr0[1], clr0[2]);
        }
    }

    mesh.CreatePointsAttr(pxr::VtValue{Points});
    mesh.CreateFaceVertexCountsAttr(pxr::VtValue{FaceVertexCounts});
    mesh.CreateFaceVertexIndicesAttr(pxr::VtValue{FaceVertexIndices});
    mesh.CreateDisplayColorAttr(pxr::VtValue{DisplayColor});

    mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeomTokens->vertex);
}

bool zenovis::StageManager::load_objects(const std::map<std::string, std::shared_ptr<zeno::IObject>> &objs) {
    auto ins = zenoObjects.insertPass();
    bool inserted = false;

    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {

            std::string p_path, p_type;
            PrimInfo primInfo;
            obj->userData().has("P_Path") ? p_path = obj->userData().get2<std::string>("P_Path") : p_path = "";
            obj->userData().has("P_Type") ? p_type = obj->userData().get2<std::string>("P_Type") : p_type = "";
            primInfo.pPath = p_path; primInfo.iObject = obj;
            zeno::log_info("USD: StageManager Emplace {}, P_Type {}, P_Path {}", key, p_type, p_path);

            if(p_type == _tokens->UsdGeomMesh.GetString()){
                _UsdGeomMesh(primInfo);
            }else if(p_type == _tokens->UsdLuxDiskLight.GetString()){

            }else{
                zeno::log_info("USD: Unsupported type {}", p_type);
            }

            ins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }

    //std::string stageString;
    //stagePtr->ExportToString(&stageString);
    //std::cout << "USD: Stage " << std::endl << stageString << std::endl;
    stagePtr->Save();

    return inserted;
}

}