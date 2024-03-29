#pragma once

#include "zenonode.h"

/*** USD headers ***/
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

/*** Zeno headers ***/
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
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
	void _emitCreateSphereNode(ZENO_HANDLE targetGraph);
	void _emitCreateCapsuleNode(ZENO_HANDLE targetGraph);
	void _emitCreateCubeNode(ZENO_HANDLE targetGraph);
	void _emitCreateCylinderNode(ZENO_HANDLE targetGraph);
	void _emitCreateConeNode(ZENO_HANDLE targetGraph);
	void _emitCreatePlaneNode(ZENO_HANDLE targetGraph);
	void _emitPrimitiveTransformNode(ZENO_HANDLE targetGraph);

	std::string mUSDPath;
	std::string mPrimPath;
	pxr::UsdPrim mUSDPrim;

private slots:
	void _onEvalClicked();
};
