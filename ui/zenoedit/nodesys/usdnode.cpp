#ifdef ZENO_ENABLE_USD
#include "usdnode.h"
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


#include <zeno/utils/eulerangle.h>

#include <iostream>

const static float NODE_LEVEL_OFFSET_X = 1000.0f;
const static float NODE_LEVEL_OFFSET_Y = 800.0f;
const static float NODE_INTERNAL_OFFSET_Y = 120.0f;
const static float NODE_INTERNAL_OFFSET_X = 300.0f;

NodePos NodePos::nextInternalPos() const {
	return {x + NODE_INTERNAL_OFFSET_X, y};
}

NodePos NodePos::nextLevelPos() const {
	return {x + NODE_LEVEL_OFFSET_X, y};
}

NodePos NodePos::nextBrotherPos() const {
	return {x, y + NODE_LEVEL_OFFSET_Y};
}

float _parseScalar(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& value) {
	float ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		ret = static_cast<float>(value.Get<double>());
	}
	else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		ret = value.Get<float>();
	}
	else {
		pxr::GfHalf v = value.Get<pxr::GfHalf>();
		ret = float(v);
	}
	return ret;
}

zeno::vec3f _parseVector3(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& vecValue) {
	zeno::vec3f ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		pxr::GfVec3d vec = vecValue.Get<pxr::GfVec3d>();
		ret = zeno::vec3f{ (float)vec[0], (float)vec[1], (float)vec[2] };
	}
	else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		pxr::GfVec3f vec = vecValue.Get<pxr::GfVec3f>();
		ret = { vec[0], vec[1], vec[2] };
	}
	else {
		pxr::GfVec3h vec = vecValue.Get<pxr::GfVec3h>();
		ret = { (float)vec[0], (float)vec[1], (float)vec[2] };
	}
	return ret;
}

zeno::vec4f _parseQuatVector(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& vecValue) {
	zeno::vec4f ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		pxr::GfQuatd quad = vecValue.Get<pxr::GfQuatd>();
		pxr::GfVec3d vec = quad.GetImaginary();
		ret = { float(vec[0]), float(vec[1]), float(vec[2]), float(quad.GetReal()) };
	}
	else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		pxr::GfQuatf quad = vecValue.Get<pxr::GfQuatf>();
		pxr::GfVec3f vec = quad.GetImaginary();
		ret = { vec[0], vec[1], vec[2], quad.GetReal() };
	}
	else {
		pxr::GfQuath quad = vecValue.Get<pxr::GfQuath>();
		pxr::GfVec3h vec = quad.GetImaginary();
		ret = { float(vec[0]), float(vec[1]), float(vec[2]), float(quad.GetReal()) };
	}
	return ret;
}

std::string getOutNameOfGeoNode(ZENO_HANDLE targetGraph, ZENO_HANDLE node) {
	/*
	* for CreateXXX node, use 'prim' as output name
	* for PrimitiveTransform, use 'outPrim' as output name
	*/
	std::string name;
	Zeno_GetName(targetGraph, node, name);
	name = (name == "PrimitiveTransform") ? "outPrim" : "prim";
	return name;
}

EvalUSDPrim::EvalUSDPrim(const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent) {
	;
}

EvalUSDPrim::~EvalUSDPrim() {
	;
}

ZGraphicsLayout* EvalUSDPrim::initCustomParamWidgets() {
	ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
	ZGraphicsLayout* pHLayoutNode = new ZGraphicsLayout(true);
	ZSimpleTextItem* pNodeItem = new ZSimpleTextItem("Eval");

	pNodeItem->setBrush(m_renderParams.socketClr.color());
	pNodeItem->setFont(m_renderParams.socketFont);
	pNodeItem->updateBoundingRect();
	pHLayoutNode->addItem(pNodeItem);

	ZenoParamPushButton* pNodeBtn = new ZenoParamPushButton("Generate", -1, QSizePolicy::Expanding);
	pHLayoutNode->addItem(pNodeBtn);
	connect(pNodeBtn, SIGNAL(clicked()), this, SLOT(_onEvalClicked()));

	_param_ctrl paramNode;
	paramNode.param_name = pNodeItem;
	paramNode.param_control = pNodeBtn;
	paramNode.ctrl_layout = pHLayoutNode;
	addParam(paramNode);

	pHLayout->addLayout(pHLayoutNode);

	return pHLayout;
}

void EvalUSDPrim::_setNodePos(ZENO_HANDLE main, ZENO_HANDLE node, const NodePos& pos) {
	Zeno_SetPos(main, node, {pos.x, pos.y});
	mLastNodePos = pos;
}

void EvalUSDPrim::_getNodeInputs() {
	ZENO_HANDLE hGraph = Zeno_GetGraph("main");
	ZENO_HANDLE curNode = index().internalId();
	ZVARIANT usdPath;
	ZVARIANT primPath;
	ZVARIANT isRecursive;
	std::string type;

	Zeno_GetInputDefl(hGraph, curNode, "USDDescription", usdPath, type);
	mUSDPath = std::get<std::string>(usdPath);

	Zeno_GetInputDefl(hGraph, curNode, "primPath", primPath, type);
	mPrimPath = std::get<std::string>(primPath);

	Zeno_GetInputDefl(hGraph, curNode, "isRecursive", isRecursive, type);
	mIsRecursive = std::get<bool>(isRecursive);
}

void EvalUSDPrim::_onEvalFinished() {
	mUSDPath = "";
	mPrimPath = "";
	mIsRecursive = false;
	mLastNodePos = {0.0f, 0.0f};

	zeno::log_info("USD prim evaling finished.");
}

// This function will parse the given USD prim, convert it to zeno node graph and return the last node handle of the graph
ZENO_HANDLE EvalUSDPrim::_parsePrim(std::any _stage, std::any _prim, const NodePos& nodePos) {
	auto stage = std::any_cast<pxr::UsdStageRefPtr>(_stage);
	auto usdPrim = std::any_cast<pxr::UsdPrim>(_prim);

	std::string primType = "";
	primType = usdPrim.GetTypeName();
	if (primType.empty()) {
		zeno::log_warn("failed to read prim type of " + mPrimPath);
		return 0;
	}

	auto mainGraph = Zeno_GetGraph("main");
	if (mainGraph == 0) {
		zeno::log_warn("failed to get main graph");
		return 0;
	}

	ZENO_HANDLE primNode = 0;
	if (primType == "Sphere") {
		primNode = _emitCreateSphereNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Capsule") {
		primNode = _emitCreateCapsuleNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Cube") {
		primNode = _emitCreateCubeNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Cylinder") {
		primNode = _emitCreateCylinderNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Cone") {
		primNode = _emitCreateConeNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Plane") {
		primNode = _emitCreatePlaneNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "Mesh") {
		primNode = _emitImportUSDMeshNode(usdPrim, mainGraph, nodePos);
	}
	else if (primType == "CylinderLight") {
		// create a geo node and a light node, then connect them
		auto geoNode = _emitCreateCylinderNode(usdPrim, mainGraph, nodePos, true);
		auto lightNode = _emitLightNode(usdPrim, mainGraph, mLastNodePos.nextInternalPos(), "Diffuse", "TriangleMesh");
		if (geoNode && lightNode) {
			Zeno_AddLink(mainGraph, geoNode, getOutNameOfGeoNode(mainGraph, geoNode), lightNode, "prim");
			primNode = lightNode;
		}
		else {
			zeno::log_warn("failed to create CreateCylinder or LightNode while evaling prim " + mPrimPath);
			primNode = geoNode + lightNode;
		}
	}
	else if (primType == "DiskLight") {
		/*
		* Details of DiskLight from USD doc:
		* Light emitted from one side of a circular disk.
		* The disk is centered in the XY plane and emits light along the -Z axis.
		*/
		auto geoNode = _emitCreateDiskNode(usdPrim, mainGraph, nodePos, true);
		auto lightNode = _emitLightNode(usdPrim, mainGraph, mLastNodePos.nextInternalPos(), "Diffuse", "TriangleMesh");
		if (geoNode && lightNode) {
			Zeno_AddLink(mainGraph, geoNode, getOutNameOfGeoNode(mainGraph, geoNode), lightNode, "prim");
			primNode = lightNode;
		}
		else {
			zeno::log_warn("failed to create CreateDisk or LightNode while evaling prim " + mPrimPath);
			primNode = geoNode + lightNode;
		}
	}
	else if (primType == "DomeLight") {
		// I think this type is not fully supported yet
		auto geoNode = _emitCreateSphereNode(usdPrim, mainGraph, nodePos, true);
		auto lightNode = _emitLightNode(usdPrim, mainGraph, mLastNodePos.nextInternalPos(), "Diffuse", "TriangleMesh");
		if (geoNode && lightNode) {
			Zeno_SetInputDefl(mainGraph, geoNode, "isFlipFace", true); // emit inside the sphere
			Zeno_AddLink(mainGraph, geoNode, "prim", lightNode, "prim");
			primNode = lightNode;
		}
		else {
			zeno::log_warn("failed to create CreateSphere or LightNode while evaling prim " + mPrimPath);
			primNode = geoNode + lightNode;
		}
	}
	else if (primType == "DistantLight") {
		// this type is not support yet
		zeno::log_warn("DistantLight is not supported by zeno yet.");
	}
	else if (primType == "RectLight") {
		/*
		* Description from USD doc about RectLight:
		* Light emitted from one side of a rectangle.
		* The rectangle is centered in the XY plane and emits light along the - Z axis.
		* The rectangle is 1 unit in length in the X and Y axis.
		* In the default position, a texture file's min coordinates should be at (+X, +Y) and max coordinates at (-X, -Y).
		*/
		// this type is not fully supported yet
		auto geoNode = _emitCreatePlaneNode(usdPrim, mainGraph, nodePos, true);
		auto lightNode = _emitLightNode(usdPrim, mainGraph, mLastNodePos.nextInternalPos(), "Diffuse", "TriangleMesh");
		if (geoNode && lightNode) {
			Zeno_AddLink(mainGraph, geoNode, "prim", lightNode, "prim");
			primNode = lightNode;
		}
		else {
			zeno::log_warn("failed to create CreatePlane or LightNode while evaling prim " + mPrimPath);
			primNode = geoNode + lightNode;
		}
	}
	else if (primType == "SphereLight") {
		// this type is not fully supported yet
		auto geoNode = _emitCreateSphereNode(usdPrim, mainGraph, nodePos, true);
		auto lightNode = _emitLightNode(usdPrim, mainGraph, mLastNodePos.nextInternalPos(), "Diffuse", "TriangleMesh");
		if (geoNode && lightNode) {
			Zeno_AddLink(mainGraph, geoNode, getOutNameOfGeoNode(mainGraph, geoNode), lightNode, "prim");
			primNode = lightNode;
		}
		else {
			zeno::log_warn("failed to create CreateSphere or LightNode while evaling prim " + mPrimPath);
			primNode = geoNode + lightNode;
		}
	}
	else if (primType == "Camera") {
		primNode = _emitCameraNode(usdPrim, mainGraph, nodePos);
	}

	auto finalNode = _emitPrimitiveTransformNodes(usdPrim, mainGraph, primNode == 0 ? nodePos : mLastNodePos.nextInternalPos(), primNode);
	Zeno_SetView(mainGraph, finalNode, true);

	return finalNode;
}

// parse the entire prim tree of the given prim
ZENO_HANDLE EvalUSDPrim::_dfsParse(ZENO_HANDLE fatherNode, std::any scene, std::any prim, const NodePos& nodePos) {
	auto usdPrim = std::any_cast<pxr::UsdPrim>(prim);

	auto node = _parsePrim(scene, usdPrim, nodePos);
	if (node == 0) {
		zeno::log_warn("failed to parse USD prim " + usdPrim.GetPath().GetString());
		return 0;
	}

	NodePos childPos = mLastNodePos.nextLevelPos();
	auto range = usdPrim.GetChildren();
	for (auto& child : range) {
		auto childNode = _dfsParse(node, scene, child, childPos);

		if (childNode == 0) {
			zeno::log_warn("failed to parse USD prim node " + child.GetPath().GetString());
		}
		else {
			// align with last child's node tree
			childPos = childPos.nextBrotherPos();
		}
	}

	return node;
}

void EvalUSDPrim::_onEvalClicked() {
	_getNodeInputs();
	if (mUSDPath.empty() || mPrimPath.empty()) {
		zeno::log_warn("[EvalUSDPrim] [Warn] found USD path or prim path is empty");
		return;
	}

	ZENO_HANDLE mainGraph = Zeno_GetGraph("main");
	if (mainGraph == 0) {
		zeno::log_error("failed to get main graph");
		return;
	}

	auto stage = pxr::UsdStage::Open(mUSDPath);
	if (!stage) {
		zeno::log_warn("failed to load usd stage " + mUSDPath);
		return;
	}

	auto usdPrim = stage->GetPrimAtPath(pxr::SdfPath(mPrimPath));
	if (!usdPrim.IsValid()) {
		zeno::log_warn("failed to load usd prim at " + mPrimPath);
		return;
	}

	std::pair<float, float> rootPos;
	auto myNode = ZenoNode::index().internalId();
	Zeno_GetPos(mainGraph, myNode, rootPos);
	rootPos.first += NODE_LEVEL_OFFSET_X; // position of the first node

	if (mIsRecursive) {
		_dfsParse(0, stage, usdPrim, {rootPos.first, rootPos.second});
	}
	else {
		_parsePrim(stage, usdPrim, { rootPos.first, rootPos.second});
	}

	_onEvalFinished();
}

ZENO_HANDLE EvalUSDPrim::_emitCreateSphereNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, bool isLightGeo) {
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	double radius;
	if (isLightGeo) { // create sphere for geometry light
		pxr::UsdAttribute attr;
		pxr::VtValue val;
		usdPrim.GetAttribute(pxr::TfToken("light:shaderId")).Get(&val);
		if (val.Get<pxr::TfToken>().GetString() == "DomeLight") {
			attr = usdPrim.GetAttribute(pxr::TfToken("guideRadius"));
		} else { // SphereLight
			attr = usdPrim.GetAttribute(pxr::TfToken("inputs:radius"));
		}
		attr.Get(&val);
		radius = static_cast<double>(val.Get<float>());
	} else {
		auto radiusAttr = usdPrim.GetAttribute(pxr::TfToken("radius"));
		radiusAttr.Get(&radius);
	}

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateSphere");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateSphere node");
		return 0;
	}
	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	_setNodePos(targetGraph, newNode, nodePos);

	return newNode;
}

// TODO: we need a CreateCapsule node
ZENO_HANDLE EvalUSDPrim::_emitCreateCapsuleNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos){
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	pxr::UsdAttribute attr;
	pxr::VtValue attrValue;
	char axis;
	double radius;
	double height;
	attr = usdPrim.GetAttribute(pxr::TfToken("axis"));
	if (!attr.HasValue()) return 0;
	attr.Get(&attrValue);
	axis = attrValue.Get<pxr::TfToken>().GetString()[0];

	attr = usdPrim.GetAttribute(pxr::TfToken("radius"));
	attr.Get(&radius);

	attr = usdPrim.GetAttribute(pxr::TfToken("height"));
	attr.Get(&height);

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCapsule");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateCapsule node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	_setNodePos(targetGraph, newNode, nodePos);

	if (axis != 'Y') {
		auto transNode = Zeno_AddNode(targetGraph, "PrimitiveTransform");
		if (transNode == 0) {
			std::cout << "failed to create PrimitiveTransform for USD capsule" << std::endl;
		} else {
			Zeno_AddLink(targetGraph, newNode, "prim", transNode, "prim");
			if (axis == 'X') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(0.0f, 0.0f, -90.0f));
			} else if (axis == 'Z') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(90.0f, 0.0f, 0.0f));
			} else {} // ??
			_setNodePos(targetGraph, transNode, nodePos.nextInternalPos());
			return transNode;
		}
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateCubeNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos){
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	pxr::UsdAttribute attr = usdPrim.GetAttribute(pxr::TfToken("size"));
	double size;
	attr.Get(&size);

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCube");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateCube node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "size", size);
	_setNodePos(targetGraph, newNode, nodePos);

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateCylinderNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, bool isLightGeo){
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	char axis;
	double radius;
	double height;
	pxr::UsdAttribute attr;
	pxr::VtValue attrValue;
	if (isLightGeo) {
		axis = 'X';
	} else {
		attr = usdPrim.GetAttribute(pxr::TfToken("axis"));
		if (!attr.HasValue()) return 0;
		attr.Get(&attrValue);
		axis = attrValue.Get<pxr::TfToken>().GetString()[0];
	}

	if (isLightGeo) {
		attr = usdPrim.GetAttribute(pxr::TfToken("inputs:radius")); // CylinderLight
		attr.Get(&attrValue);
		radius = static_cast<double>(attrValue.Get<float>());
	} else {
		attr = usdPrim.GetAttribute(pxr::TfToken("radius"));
		attr.Get(&radius);
	}

	if (isLightGeo) {
		attr = usdPrim.GetAttribute(pxr::TfToken("inputs:length")); // in CylinderLight, height is called 'length' :)
		attr.Get(&attrValue);
		height = static_cast<double>(attrValue.Get<float>());
	} else {
		attr = usdPrim.GetAttribute(pxr::TfToken("height"));
		attr.Get(&height);
	}

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCylinder");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateCylinder node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	_setNodePos(targetGraph, newNode, nodePos);

	if (axis != 'Y') {
		auto transNode = Zeno_AddNode(targetGraph, "PrimitiveTransform");
		if (transNode == 0) {
			std::cout << "failed to create PrimitiveTransform for USD Cylinder" << std::endl;
			return newNode;
		}

		Zeno_AddLink(targetGraph, newNode, "prim", transNode, "prim");
		if (axis == 'X') {
			Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(0.0f, 0.0f, -90.0f));
		} else if (axis == 'Z') {
			Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(90.0f, 0.0f, 0.0f));
		} else {} // ??
		_setNodePos(targetGraph, transNode, nodePos.nextInternalPos());
		return transNode;
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateConeNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos){
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	char axis;
	double radius;
	double height;
	pxr::UsdAttribute attr;
	pxr::VtValue attrValue;
	attr = usdPrim.GetAttribute(pxr::TfToken("axis"));
	if (!attr.HasValue()) return 0;
	attr.Get(&attrValue);
	axis = attrValue.Get<pxr::TfToken>().GetString()[0];

	attr = usdPrim.GetAttribute(pxr::TfToken("radius"));
	attr.Get(&radius);

	attr = usdPrim.GetAttribute(pxr::TfToken("height"));
	attr.Get(&height);

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCone");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateCone node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	_setNodePos(targetGraph, newNode, nodePos);

	if (axis != 'Y') {
		auto transNode = Zeno_AddNode(targetGraph, "PrimitiveTransform");
		if (transNode == 0) {
			std::cout << "failed to create PrimitiveTransform for USD cone" << std::endl;
		} else {
			Zeno_AddLink(targetGraph, newNode, "prim", transNode, "prim");
			if (axis == 'X') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(0.0f, 0.0f, -90.0f));
			} else if (axis == 'Z') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(90.0f, 0.0f, 0.0f));
			} else {} // ??
			_setNodePos(targetGraph, transNode, nodePos.nextInternalPos());
			return transNode;
		}
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreatePlaneNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, bool isLightGeo){
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	char axis;
	double height, width;
	pxr::UsdAttribute attr;
	pxr::VtValue val;
	if (isLightGeo) { // RectLight
		axis = 'Z';

		attr = usdPrim.GetAttribute(pxr::TfToken("inputs:height"));
		attr.Get(&val);
		height = static_cast<double>(val.Get<float>());

		attr = usdPrim.GetAttribute(pxr::TfToken("inputs:width"));
		attr.Get(&val);
		width = static_cast<double>(val.Get<float>());
	} else { // Plane
		attr = usdPrim.GetAttribute(pxr::TfToken("axis"));
		if (!attr.HasValue()) return 0;
		attr.Get(&val);
		axis = val.Get<pxr::TfToken>().GetString()[0];

		attr = usdPrim.GetAttribute(pxr::TfToken("length"));
		attr.Get(&height);

		attr = usdPrim.GetAttribute(pxr::TfToken("width"));
		attr.Get(&width);
	}

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreatePlane");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreatePlane node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "scaleSize", zeno::vec3f(width, 1.0f, height));
	// Yeah, we don't need to add the PrimitiveTransform node :)
	if (axis == 'X') {
		Zeno_SetInputDefl(targetGraph, newNode, "rotate", zeno::vec3f(90.0f, 0.0f, 0.0f));
	} else if (axis == 'Z') {
		Zeno_SetInputDefl(targetGraph, newNode, "rotate", isLightGeo ? zeno::vec3f(-90.0f, 180.0f, 0.0f) : zeno::vec3f(0.0f, 0.0f, 90.0f));
	}

	Zeno_SetInputDefl(targetGraph, newNode, "size", 1.0f);
	_setNodePos(targetGraph, newNode, nodePos);

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateDiskNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, bool isLightGeo) {
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	char axis = 'Z';
	if (isLightGeo) {
		axis = 'Z'; // disk light emit towards -Z
	} else {
		; // TODO
	}

	float radius;
	if (isLightGeo) {
		usdPrim.GetAttribute(pxr::TfToken("inputs:radius")).Get(&radius);
	} else {
		radius = 0.0f; // TODO
	}

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateDisk");
	if (newNode == 0) {
		zeno::log_error("failed to emit CreateDisk node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	_setNodePos(targetGraph, newNode, nodePos);

	if (axis != 'Y') {
		auto transNode = Zeno_AddNode(targetGraph, "PrimitiveTransform");
		if (!transNode) {
			zeno::log_error("failed to create node: PrimitiveTransform");
		} else {
			Zeno_AddLink(targetGraph, newNode, "prim", transNode, "prim");
			if (axis == 'X') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(0.0f, 0.0f, isLightGeo ? 90.0f : -90.0f));
			} else if (axis == 'Z') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(isLightGeo ? -90.0f : 90.0f, 0.0f, 0.0f));
			} else {} // ??
			_setNodePos(targetGraph, transNode, nodePos.nextInternalPos());
			return transNode;
		}
	}
	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitImportUSDMeshNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos) {
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "ImportUSDMesh");
	if (newNode == 0) {
		zeno::log_error("failed to emit ImportUSDMesh node");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "USDDescription", mUSDPath);
	Zeno_SetInputDefl(targetGraph, newNode, "primPath", mPrimPath);
	_setNodePos(targetGraph, newNode, nodePos);

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitLightNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, const std::string& lightType, const std::string& shapeType) {
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	// read parameter from USD light prim
	auto light = pxr::UsdLuxLightAPI(usdPrim);
	float exposure;
	float intensity;
	zeno::vec3f color(0.0f);
	pxr::UsdAttribute attr;
	pxr::VtValue attrValue;

	attr = light.GetIntensityAttr();
	attr.Get(&intensity);
	attr = light.GetExposureAttr();
	attr.Get(&exposure);

	// deal with light color
	ZENO_HANDLE tempNode = 0;
	bool enableTemperature = false;
	light.GetEnableColorTemperatureAttr().Get<bool>(&enableTemperature);
	if (enableTemperature) {
		float temp;
		attr = light.GetColorTemperatureAttr();
		attr.Get(&temp);

		tempNode = Zeno_AddNode(targetGraph, "ColorTemperatureToRGB");
		if (!tempNode) {
			zeno::log_warn("failed to create node: ColorTemperatureToRGB");
		}
		else {
			Zeno_SetInputDefl(targetGraph, tempNode, "temerature", temp);
			_setNodePos(targetGraph, tempNode, nodePos);
		}
	}
	else {
		attr = light.GetColorAttr();
		pxr::GfVec3f _col;
		attr.Get(&_col);
		color = { _col[0], _col[1], _col[2] };
	}

	// deal with texture in DomeLight or RectLight
	auto texAttr = usdPrim.GetAttribute(pxr::TfToken("inputs:texture:file"));
	std::string texturePath = "";
	if (texAttr.HasValue()) {
		pxr::SdfAssetPath texPath;
		texAttr.Get(&texPath);
		texturePath = texPath.GetResolvedPath();
	}

	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "LightNode");
	if (newNode == 0) {
		zeno::log_warn("failed to create node: LightNode");
		if (tempNode) {
			Zeno_DeleteNode(targetGraph, tempNode);
		}
		return 0;
	}

	// final setup
	Zeno_SetInputDefl(targetGraph, newNode, "intensity", intensity);
	Zeno_SetInputDefl(targetGraph, newNode, "exposure", exposure);
	/*
	* shape: Plane Ellipse Sphere Point TriangleMesh
	* type: Diffuse Direction IES Spot Projector
	*/
	Zeno_SetInputDefl(targetGraph, newNode, "type", lightType);
	Zeno_SetInputDefl(targetGraph, newNode, "shape", shapeType);

	Zeno_SetInputDefl(targetGraph, newNode, "texturePath", texturePath);

	if (enableTemperature && tempNode) {
		Zeno_AddLink(targetGraph, tempNode, "color", newNode, "color");
		_setNodePos(targetGraph, newNode, nodePos.nextInternalPos());
	}
	else {
		Zeno_SetInputDefl(targetGraph, newNode, "color", color);
		_setNodePos(targetGraph, newNode, nodePos);
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCameraNode(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos) {
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	auto camera = pxr::UsdGeomCamera(usdPrim);

	pxr::GfVec2f clipRange;
	camera.GetClippingRangeAttr().Get(&clipRange);

	float focalLength;
	camera.GetFocalLengthAttr().Get(&focalLength);

	// use vertical aperture, ignore horizontal for now
	float aperture;
	camera.GetVerticalApertureAttr().Get(&aperture);
	// TODO: aperture offset

	// TODO: fov
	// TODO: projection

	auto newNode = Zeno_AddNode(targetGraph, "MakeCamera");
	if (newNode == 0) {
		zeno::log_warn("failed to create node MakeCamera");
		return 0;
	}

	Zeno_SetInputDefl(targetGraph, newNode, "near", clipRange[0]);
	Zeno_SetInputDefl(targetGraph, newNode, "far", clipRange[1]);
	Zeno_SetInputDefl(targetGraph, newNode, "focalPlaneDistance", focalLength);
	Zeno_SetInputDefl(targetGraph, newNode, "aperture", aperture);
	_setNodePos(targetGraph, newNode, nodePos);

	return newNode;
}

// return root node of the entire transform link
ZENO_HANDLE EvalUSDPrim::_emitPrimitiveTransformNodes(std::any prim, ZENO_HANDLE targetGraph, const NodePos& nodePos, ZENO_HANDLE lastNode) {
	pxr::UsdPrim usdPrim = std::any_cast<pxr::UsdPrim>(prim);
	auto xform = pxr::UsdGeomXform(usdPrim);
	bool resetXformStack;
	auto xformOps = xform.GetOrderedXformOps(&resetXformStack);

	NodePos curNodePos = nodePos;
	// traverse and parse xformOps into PrimitiveTransform nodes
	for (auto& op: xformOps) {
		auto opType = op.GetOpType();
		if (opType == pxr::UsdGeomXformOp::TypeInvalid) {
			zeno::log_warn("found invalid op type while evaling xform ops");
			continue;
		}
		auto precision = op.GetPrecision();
		pxr::VtValue vecValue;
		ZVARIANT finalVec;

		op.Get(&vecValue);
		if (opType == pxr::UsdGeomXformOp::TypeTransform) {
			finalVec = op.GetName(); // we don't parse matrix here, let ImportUSDPrimMatrix node do it
		} else if (opType == pxr::UsdGeomXformOp::TypeTranslate) {
			finalVec = _parseVector3(precision, vecValue);
		} else if (opType == pxr::UsdGeomXformOp::TypeOrient) {
			finalVec = _parseQuatVector(precision, vecValue);
		} else if (opType == pxr::UsdGeomXformOp::TypeScale){
			finalVec = _parseVector3(precision, vecValue);
		} else if (opType >= pxr::UsdGeomXformOp::TypeRotateXYZ && opType <= pxr::UsdGeomXformOp::TypeRotateZYX) {
			finalVec = _parseVector3(precision, vecValue);
		} else if (opType >= pxr::UsdGeomXformOp::TypeRotateX && opType <= pxr::UsdGeomXformOp::TypeRotateZ) {
			finalVec = _parseScalar(precision, vecValue);
			float euler = std::get<float>(finalVec);
			if (opType == pxr::UsdGeomXformOp::TypeRotateX) {
				finalVec = zeno::vec3f(euler, 0.0f, 0.0f);
			} else if (opType == pxr::UsdGeomXformOp::TypeRotateY) {
				finalVec = zeno::vec3f(0.0f, euler, 0.0f);
			} else { // Z axis
				finalVec = zeno::vec3f(0.0f, 0.0f, euler);
			}
		}

		ZENO_HANDLE newNode = _makeTransformNode(targetGraph, curNodePos, opType, finalVec);
		if (newNode == 0) {
			continue;
		}

		if (lastNode != 0) {
			Zeno_AddLink(targetGraph, lastNode, getOutNameOfGeoNode(targetGraph, lastNode), newNode, "prim");
		}
		lastNode = newNode;
		curNodePos = curNodePos.nextInternalPos();
	}

	return lastNode;
}

ZENO_HANDLE EvalUSDPrim::_makeTransformNode(ZENO_HANDLE main, const NodePos& nodePos, std::any transT, const ZVARIANT& transValue) {
	auto transType = std::any_cast<const pxr::UsdGeomXformOp::Type&>(transT);
	ZENO_HANDLE newNode = Zeno_AddNode(main, "PrimitiveTransform"); 
	if (newNode == 0) {
		zeno::log_warn("failed to create node named PrimitiveTransform");
		return 0;
	}

	if (transType == pxr::UsdGeomXformOp::TypeTranslate) {
		Zeno_SetInputDefl(main, newNode, "translation", std::get<zeno::vec3f>(transValue));
	} else if (transType == pxr::UsdGeomXformOp::TypeScale) {
		Zeno_SetInputDefl(main, newNode, "scaling", std::get<zeno::vec3f>(transValue));
	} else if (transType >= pxr::UsdGeomXformOp::TypeRotateX && transType <= pxr::UsdGeomXformOp::TypeRotateZ) {
		// only need to rotate one axis, so we don't care rotation order here
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateXYZ) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("XYZ"));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateXZY) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("XZY"));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateYXZ) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("YXZ"));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateYZX) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("YZX"));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateZXY) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("ZXY"));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateZYX) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transValue));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", std::string("ZYX"));
	} else if (transType == pxr::UsdGeomXformOp::TypeOrient) {
		Zeno_SetInputDefl(main, newNode, "quatRotation", std::get<zeno::vec4f>(transValue));
	} else if (transType == pxr::UsdGeomXformOp::TypeTransform) {
		// add node to parse matrix and link to the input
		auto importNode = Zeno_AddNode(main, "ImportUSDPrimMatrix");
		if (importNode == 0) {
			zeno::log_warn("failed to create node: ImportUSDPrimMatrix");
			return newNode;
		}

		const std::string& attrName = std::get<std::string>(transValue);

		Zeno_SetInputDefl(main, importNode, "USDDescription", mUSDPath);
		Zeno_SetInputDefl(main, importNode, "primPath", mPrimPath);
		Zeno_SetInputDefl(main, importNode, "attribute", attrName);
		Zeno_AddLink(main, importNode, "Matrix", newNode, "Matrix");
	} else {
		Zeno_DeleteNode(main, newNode);
		return 0;
	}

	_setNodePos(main, newNode, nodePos);

	return newNode;
}
#endif