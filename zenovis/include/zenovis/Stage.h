#pragma once

// Include first for avoid error WinSock.h has already been included
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <boost/predef/os.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/js/json.h>
#include <pxr/base/plug/plugin.h>
#include <pxr/base/tf/fileUtils.h>
#include <pxr/pxr.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usd/inherits.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>

#include <zeno/core/IObject.h>

PXR_NAMESPACE_USING_DIRECTIVE

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

struct ConfigurationInfo{
    std::string cRepo = "http://test1:12345@192.168.3.11:8000/r/zeno_usd_test.git";
    std::string cPath = "C:/Users/Public/zeno_usd_test";
    std::string cServer = "192.168.3.11";
};

struct PrimInfo{
    std::string pPath;
    std::shared_ptr<zeno::IObject> iObject;
};

struct ZenoStage{
    UsdStageRefPtr cStagePtr;
    UsdStageRefPtr fStagePtr;
    UsdStageRefPtr sStagePtr;
    ConfigurationInfo confInfo;

    std::string pathEnv;

    ZenoStage();

    void update();

    int Convert2UsdGeomMesh(const PrimInfo& primInfo);

    void CreateUSDHierarchy(const SdfPath &path)
    {
        if (path == SdfPath::AbsoluteRootPath())
            return;
        CreateUSDHierarchy(path.GetParentPath());
        UsdGeomXform::Define(cStagePtr, path);
    }

};