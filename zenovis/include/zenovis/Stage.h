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

struct ConfigurationInfo{
    std::string cRepo = "http://test1:12345@192.168.2.106:8000/r/zeno_usd_test.git";
    std::string cPath = "C:/Users/Public/zeno_usd_test";
    std::string cServer = "192.168.2.106";
};

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

    ConfigurationInfo confInfo;
    SdfLayerRefPtr composLayer;
    std::map<SdfPath, std::shared_ptr<zeno::PrimitiveObject>> convertedObject;

    ZenoStage();

    void update();

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