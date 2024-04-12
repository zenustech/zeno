#pragma once

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

	ZENO_HANDLE _emitPrimitiveTransformNodes(std::any, ZENO_HANDLE targetGraph, ZENO_HANDLE lastNode);

	ZENO_HANDLE _makeTransformNode(ZENO_HANDLE mainGraph, const std::pair<float, float>&, std::any transType, const ZVARIANT& transVec);

	std::string mUSDPath;
	std::string mPrimPath;

private slots:
	void _onEvalClicked();
};
