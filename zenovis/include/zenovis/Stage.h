#pragma once

// Include first for avoid error WinSock.h has already been included
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/predef/os.h>

#include <pxr/pxr.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/js/json.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/tf/fileUtils.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usd/inherits.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>

#include <zeno/core/IObject.h>
#include <zeno/types/UserData.h>
#include <zeno/types/PrimitiveObject.h>

#include <chrono>

#include "zenovis/StageCommon.h"

#define TIMER_START(NAME) \
    auto start_##NAME = std::chrono::high_resolution_clock::now();

#define TIMER_END(NAME) \
    auto stop_##NAME = std::chrono::high_resolution_clock::now(); \
    auto duration_##NAME = std::chrono::duration_cast<std::chrono::microseconds>(stop_##NAME - start_##NAME); \
    std::cout << "USD: Timer " << #NAME << " "                          \
              << duration_##NAME.count()*0.001f << " Milliseconds" << std::endl;


PXR_NAMESPACE_USING_DIRECTIVE

TF_DEFINE_PRIVATE_TOKENS(
    _primTokens,
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

    (UsdGeomCube)
    (UsdGeomSphere)
);

TF_DEFINE_PRIVATE_TOKENS(
    _typeTokens,
    // type
    (Mesh)
);

struct ZPrimInfo{
    std::string pPath;
    std::shared_ptr<zeno::IObject> iObject;
};

struct UPrimInfo{
    UsdPrim usdPrim;
    std::shared_ptr<zeno::PrimitiveObject> iObject;
};

struct ZenoStage{
    UsdStageRefPtr cStagePtr;
    UsdStageRefPtr fStagePtr;
    UsdStageRefPtr sStagePtr;

    HandleStateInfo *stateInfo;
    SdfLayerRefPtr composLayer;
    std::map<SdfPath, std::shared_ptr<zeno::PrimitiveObject>> convertedObject;

    ZenoStage();

    void init();
    void update();

    int PrintStageString(UsdStageRefPtr stage);
    int PrintLayerString(SdfLayerRefPtr layer);
    int CompositionArcsStage();
    int TraverseStageObjects(UsdStageRefPtr stage, std::map<std::string, UPrimInfo>& consis);
    int RemoveStagePrims();
    int CheckConvertConsistency(UsdPrim& prim);
    int CheckPathConflict();
    int CheckAttrVisibility(const UsdPrim& prim);
    void CreateUSDHierarchy(const SdfPath &path);

    int Convert2UsdGeomMesh(const ZPrimInfo& primInfo);
    int Convert2ZenoPrimitive(const UPrimInfo& primInfo);



};