#pragma once
#ifdef ZENO_ENABLE_USD
#include "zenonode.h"

/*** Zeno headers ***/
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/graphsmanagment.h>
#include <any>

struct NodePos {
	float x;
	float y;
	NodePos nextInternalPos() const;
	NodePos nextLevelPos() const;
	NodePos nextBrotherPos() const;
};

class EvalUSDPrim: public ZenoNode {
	Q_OBJECT
public:
	EvalUSDPrim(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~EvalUSDPrim();

protected:
	ZGraphicsLayout* initCustomParamWidgets() override;

private:
	void _setNodePos(ZENO_HANDLE, ZENO_HANDLE, const NodePos&);

	void _getNodeInputs();
	void _onEvalFinished();
	ZENO_HANDLE _parsePrim(/* pxr::UsdStageRefPtr */ std::any scene, /* pxr::UsdPrim */ std::any, const NodePos& nodePos);
	ZENO_HANDLE _dfsParse(ZENO_HANDLE father, /* pxr::UsdStageRefPtr */ std::any scene, /* pxr::UsdPrim */ std::any prim, const NodePos& nodePos);

	/*** Basic Geometry Node Generation ***/
	ZENO_HANDLE _emitCreateSphereNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateCapsuleNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&);
	ZENO_HANDLE _emitCreateCubeNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&);
	ZENO_HANDLE _emitCreateCylinderNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateConeNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&);
	ZENO_HANDLE _emitCreatePlaneNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&, bool isLightGeo = false);
	ZENO_HANDLE _emitCreateDiskNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&, bool isLightGeo = false);
	ZENO_HANDLE _emitImportUSDMeshNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos&);

	ZENO_HANDLE _emitLightNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos& nodePos, const std::string& lightType, const std::string& shapeType);

	ZENO_HANDLE _emitCameraNode(/* pxr::UsdPrim */ std::any, ZENO_HANDLE, const NodePos& nodePos);

	ZENO_HANDLE _emitPrimitiveTransformNodes(/* pxr::UsdPrim */ std::any, ZENO_HANDLE targetGraph, const NodePos& nodePos, ZENO_HANDLE lastNode);

	ZENO_HANDLE _makeTransformNode(ZENO_HANDLE mainGraph, const NodePos& nodePos, /* pxr::UsdGeomXformOp::Type */ std::any transType, const ZVARIANT& transValue);

	std::string mUSDPath;
	std::string mPrimPath;
	bool mIsRecursive = false;
	NodePos mLastNodePos;

private slots:
	void _onEvalClicked();
};

#endif
