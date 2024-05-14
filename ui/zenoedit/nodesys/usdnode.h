#pragma once
#ifdef ZENO_ENABLE_USD
#include "zenonode.h"

/*** Zeno headers ***/
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/graphsmanagment.h>
#include <any>

class EvalUSDPrim: public ZenoNode {
	Q_OBJECT
public:
	EvalUSDPrim(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~EvalUSDPrim();

protected:
	ZGraphicsLayout* initCustomParamWidgets() override;

private:
	void _getNodeInputs();
	void _onEvalFinished(); // clean up context
	ZENO_HANDLE _parsePrimNoXform(ZENO_HANDLE mainGraph, /* pxr::UsdStageRefPtr */ std::any scene, /* pxr::UsdPrim */ std::any);
	ZENO_HANDLE _dfsParse(ZENO_HANDLE mainGraph, /* pxr::UsdStageRefPtr */ std::any scene, /* pxr::UsdPrim */ std::any prim);
	ZENO_HANDLE _singleParse(ZENO_HANDLE mainGraph, /* pxr::UsdStageRefPtr */ std::any scene, /* pxr::UsdPrim */ std::any prim);

	/*** Basic Geometry Node Generation ***/
	ZENO_HANDLE _emitCreateSphereNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateCapsuleNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreateCubeNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreateCylinderNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateConeNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitCreatePlaneNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateDiskNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, bool isLightGeo = false);
	ZENO_HANDLE _emitImportUSDMeshNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);

	ZENO_HANDLE _emitMaterialNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);
	ZENO_HANDLE _emitSurfaceShaderNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);
	void _handleShaderInput(/* pxr::UsdPrim */ std::any, const std::string&, ZENO_HANDLE, ZENO_HANDLE, const std::string&);

	ZENO_HANDLE _emitLightNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const std::string& lightType, const std::string& shapeType);

	ZENO_HANDLE _emitCameraNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE);

	ZENO_HANDLE _emitPrimitiveTransformNodes(/* pxr::UsdPrim */ std::any, ZENO_HANDLE targetGraph, ZENO_HANDLE lastNode);

	ZENO_HANDLE _makeTransformNode(ZENO_HANDLE mainGraph, /* pxr::UsdGeomXformOp::Type */ std::any transType, const ZVARIANT& transValue);

	std::string mUSDPath;
	std::string mPrimPath;
	bool mIsRecursive = false;
private slots:
	void _onEvalClicked();
};

#endif
