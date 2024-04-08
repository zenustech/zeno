#pragma once

#include "zenonode.h"

/*** USD headers ***/
#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>

#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cone.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/plane.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformOp.h>

#include <pxr/usd/usdLux/cylinderLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/geometryLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>

/*** Zeno headers ***/
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/graphsmanagment.h>

class EvalUSDPrim: public ZenoNode {
	Q_OBJECT
public:
	EvalUSDPrim(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~EvalUSDPrim();

protected:
	ZGraphicsLayout* initCustomParamWidgets() override;

private:
	void _getNodeInputs();
	void _onEvalFinished();

	/*** Basic Geometry Node Generation ***/
	ZENO_HANDLE _emitCreateSphereNode(std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateCapsuleNode(std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreateCubeNode(std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreateCylinderNode(std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateConeNode(std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreatePlaneNode(std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateDiskNode(std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitImportUSDMeshNode(std::any, ZENO_HANDLE);

	ZENO_HANDLE _emitLightNode(std::any, ZENO_HANDLE, const std::string& lightType, const std::string& shapeType);

	ZENO_HANDLE _emitCameraNode(std::any, ZENO_HANDLE);

	void _emitPrimitiveTransformNodes(std::any, ZENO_HANDLE targetGraph, ZENO_HANDLE lastNode);

	ZENO_HANDLE _makeTransformNode(ZENO_HANDLE mainGraph, std::any transType, const ZVARIANT& transVec);

	std::string mUSDPath;
	std::string mPrimPath;

private slots:
	void _onEvalClicked();
};
