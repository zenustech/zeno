#include "usdnode.h"

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

void EvalUSDPrim::_getNodeInputs() {
	ZENO_HANDLE hGraph = Zeno_GetGraph("main");
	ZENO_HANDLE curNode = index().internalId();
	ZVARIANT usdPath;
	ZVARIANT primPath;
	std::string type;

	Zeno_GetInputDefl(hGraph, curNode, "USDDescription", usdPath, type);
	mUSDPath = std::get<std::string>(usdPath);

	Zeno_GetInputDefl(hGraph, curNode, "primPath", primPath, type);
	mPrimPath = std::get<std::string>(primPath);
}

void EvalUSDPrim::_onEvalFinished() {
	mUSDPath = "";
	mPrimPath = "";
	;
}

void EvalUSDPrim::_onEvalClicked() {
	_getNodeInputs();
	if (mUSDPath.empty() || mPrimPath.empty()) {
		std::cout << "[EvalUSDPrim] [Warn] found USD path or prim path is empty" << std::endl;
		return;
	}

	auto stage = pxr::UsdStage::Open(mUSDPath);
	if (!stage) {
		std::cout << "failed to load usd stage " << mUSDPath << std::endl;
		return;
	}

	mUSDPrim = stage->GetPrimAtPath(pxr::SdfPath(mPrimPath));
	if (!mUSDPrim.IsValid()) {
		std::cout << "failed to load usd prim at " << mPrimPath << std::endl;
		return;
	}

	std::string primType = "";
	primType = mUSDPrim.GetTypeName();
	if (primType.empty()) {
		std::cout << "failed to read prim type of " << mPrimPath << std::endl;
		return;
	}

	// TODO: remember to set prim path into its userData

	auto mainGraph = Zeno_GetGraph("main");

	ZENO_HANDLE primNode = 0;
	if (primType == "Sphere") {
		primNode = _emitCreateSphereNode(mainGraph);
	} else if (primType == "Capsule"){
		primNode = _emitCreateCapsuleNode(mainGraph);
	} else if (primType == "Cube"){
		primNode = _emitCreateCubeNode(mainGraph);
	} else if (primType == "Cylinder"){
		primNode = _emitCreateCylinderNode(mainGraph);
	} else if (primType == "Cone"){
		primNode = _emitCreateConeNode(mainGraph);
	} else if (primType == "Plane"){
		primNode = _emitCreatePlaneNode(mainGraph);
	} else if (primType == "Mesh") {
		;
	}

	_emitPrimitiveTransformNodes(mainGraph, primNode);
}

ZENO_HANDLE EvalUSDPrim::_emitCreateSphereNode(ZENO_HANDLE targetGraph) {
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateSphere");
	if (newNode == 0) {
		std::cout << "failed to emit CreateSphere node" << std::endl;
		return 0;
	}

	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	auto sphere = pxr::UsdGeomSphere(mUSDPrim);
	double radius;
	sphere.GetRadiusAttr().Get(&radius);
	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

	return newNode;
}

// TODO: we need a CreateCapsule node
ZENO_HANDLE EvalUSDPrim::_emitCreateCapsuleNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCapsule");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCapsule node" << std::endl;
		return 0;
	}
	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);
	auto capsule = pxr::UsdGeomCapsule(mUSDPrim);

	char axis;
	pxr::VtValue axisValue;
	capsule.GetAxisAttr().Get(&axisValue);
	axis = axisValue.Get<pxr::TfToken>().GetString()[0];

	double radius;
	double height;
	capsule.GetRadiusAttr().Get(&radius);
	capsule.GetHeightAttr().Get(&height);
	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

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
			Zeno_SetPos(targetGraph, transNode, { nodePos.first + 50.0f, nodePos.second });
			Zeno_SetView(targetGraph, transNode, true);
		}
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateCubeNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCube");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCube node" << std::endl;
		return 0;
	}

	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	double size;
	mUSDPrim.GetAttribute(pxr::TfToken("size")).Get(&size);
	Zeno_SetInputDefl(targetGraph, newNode, "size", size);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateCylinderNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCylinder");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCylinder node" << std::endl;
		return 0;
	}
	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	auto cylinder = pxr::UsdGeomCylinder(mUSDPrim);

	double radius;
	double height;
	cylinder.GetRadiusAttr().Get(&radius);
	cylinder.GetHeightAttr().Get(&height);

	char axis;
	pxr::VtValue axisValue;
	cylinder.GetAxisAttr().Get(&axisValue);
	axis = axisValue.Get<pxr::TfToken>().GetString()[0];

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

	if (axis != 'Y') {
		auto transNode = Zeno_AddNode(targetGraph, "PrimitiveTransform");
		if (transNode == 0) {
			std::cout << "failed to create PrimitiveTransform for USD Cylinder" << std::endl;
		} else {
			Zeno_AddLink(targetGraph, newNode, "prim", transNode, "prim");
			if (axis == 'X') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(0.0f, 0.0f, -90.0f));
			} else if (axis == 'Z') {
				Zeno_SetInputDefl(targetGraph, transNode, "eulerXYZ", zeno::vec3f(90.0f, 0.0f, 0.0f));
			} else {} // ??
			Zeno_SetPos(targetGraph, transNode, { nodePos.first + 50.0f, nodePos.second });
			Zeno_SetView(targetGraph, transNode, true);
		}
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreateConeNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCone");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCone node" << std::endl;
		return 0;
	}

	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	auto cone = pxr::UsdGeomCone(mUSDPrim);

	double radius;
	double height;
	cone.GetRadiusAttr().Get(&radius);
	cone.GetHeightAttr().Get(&height);

	char axis;
	pxr::VtValue axisValue;
	cone.GetAxisAttr().Get(&axisValue);
	axis = axisValue.Get<pxr::TfToken>().GetString()[0];

	Zeno_SetInputDefl(targetGraph, newNode, "radius", radius);
	Zeno_SetInputDefl(targetGraph, newNode, "height", height);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

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
			Zeno_SetPos(targetGraph, transNode, { nodePos.first + 50.0f, nodePos.second });
			Zeno_SetView(targetGraph, transNode, true);
		}
	}

	return newNode;
}

ZENO_HANDLE EvalUSDPrim::_emitCreatePlaneNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreatePlane");
	if (newNode == 0) {
		std::cout << "failed to emit CreatePlane node" << std::endl;
		return 0;
	}
	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	auto plane = pxr::UsdGeomPlane(mUSDPrim);

	double length, width;
	plane.GetWidthAttr().Get(&width);
	plane.GetLengthAttr().Get(&length);
	Zeno_SetInputDefl(targetGraph, newNode, "scaleSize", zeno::vec3f(length, 1.0f, width));

	char axis;
	pxr::VtValue axisValue;
	plane.GetAxisAttr().Get(&axisValue);
	axis = axisValue.Get<pxr::TfToken>().GetString()[0];
	// Yeah, we don't need to add an PrimitiveTransform node
	if (axis == 'X') {
		Zeno_SetInputDefl(targetGraph, newNode, "rotate", zeno::vec3f(90.0f, 0.0f, 0.0f));
	} else if (axis == 'Z') {
		Zeno_SetInputDefl(targetGraph, newNode, "rotate", zeno::vec3f(0.0f, 0.0f, 90.0f));
	}

	Zeno_SetInputDefl(targetGraph, newNode, "size", 1);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);

	return newNode;
}

// return root node of the entire transform link
void EvalUSDPrim::_emitPrimitiveTransformNodes(ZENO_HANDLE targetGraph, ZENO_HANDLE lastNode) {
	auto xform = pxr::UsdGeomXform(mUSDPrim);
	bool resetXformStack;
	auto xformOps = xform.GetOrderedXformOps(&resetXformStack);
	ZENO_HANDLE curNode = index().internalId();
	bool isLastNodeTransform = false;

	// get next generate position
	std::pair<float, float> nodePos;
	if (lastNode != 0) {
		Zeno_GetPos(targetGraph, lastNode, nodePos);
	} else {
		Zeno_GetPos(targetGraph, curNode, nodePos);
	}
	nodePos.first += 100.0f;

	// traverse and parse xformOps into PrimitiveTransform nodes
	for (auto& op: xformOps) {
		auto opType = op.GetOpType();
		if (opType == pxr::UsdGeomXformOp::TypeInvalid) {
			std::cout << "[EvalUSDPrim][Warn] found invalid op type while evaling xform ops" << std::endl;
			continue;
		}

		auto precision = op.GetPrecision();
		pxr::VtValue vecValue;
		ZENO_HANDLE newNode = 0;
		ZVARIANT finalVec;

		op.Get(&vecValue);
		if (opType == pxr::UsdGeomXformOp::TypeTransform) {
			continue; // we don't support matrix node yet, so skip it
		} else if (opType == pxr::UsdGeomXformOp::TypeTranslate) {
			finalVec = _parseVector3(precision, vecValue);
		} else if (opType == pxr::UsdGeomXformOp::TypeOrient) {
			finalVec = _parseVector4(precision, vecValue);
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

		newNode = _makeTransformNode(targetGraph, opType, finalVec);
		if (newNode == 0) {
			std::cout << "[EvalUSDPrim][_emitPrimitiveTransformNodes] failed to eval transform" << std::endl;
			continue;
		}

		if (lastNode != 0) {
			if (isLastNodeTransform) {
				Zeno_AddLink(targetGraph, lastNode, "outPrim", newNode, "prim");
			} else {
				// CreateXXX nodes
				Zeno_AddLink(targetGraph, lastNode, "prim", newNode, "prim");
			}
		}
		lastNode = newNode;
		isLastNodeTransform = true;

		Zeno_SetPos(targetGraph, newNode, nodePos);
		Zeno_SetView(targetGraph, newNode, true);
		nodePos.first += 100.0f;
	}
}

float EvalUSDPrim::_parseScalar(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& value) {
	float ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		ret = static_cast<float>(value.Get<double>());
	} else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		ret = value.Get<float>();
	} else {
		pxr::GfHalf v = value.Get<pxr::GfHalf>();
		ret = float(v);
	}
	return ret;
}

zeno::vec3f EvalUSDPrim::_parseVector3(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& vecValue) {
	zeno::vec3f ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		pxr::GfVec3d vec = vecValue.Get<pxr::GfVec3d>();
		ret = zeno::vec3f{ (float)vec[0], (float)vec[1], (float)vec[2] };
	} else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		pxr::GfVec3f vec = vecValue.Get<pxr::GfVec3f>();
		ret = { vec[0], vec[1], vec[2] };
	} else {
		pxr::GfVec3h vec = vecValue.Get<pxr::GfVec3h>();
		ret = { (float)vec[0], (float)vec[1], (float)vec[2] };
	}
	return ret;
}

zeno::vec4f EvalUSDPrim::_parseVector4(pxr::UsdGeomXformOp::Precision precision, const pxr::VtValue& vecValue) {
	zeno::vec4f ret;
	if (precision == pxr::UsdGeomXformOp::PrecisionDouble) {
		pxr::GfVec4d vec = vecValue.Get<pxr::GfVec4d>();
		ret = { (float)vec[0], (float)vec[1], (float)vec[2] , (float)vec[3] };
	} else if (precision == pxr::UsdGeomXformOp::PrecisionFloat) {
		pxr::GfVec4f vec = vecValue.Get<pxr::GfVec4f>();
		ret = { vec[0], vec[1], vec[2] , vec[3]};
	} else {
		pxr::GfVec4h vec = vecValue.Get<pxr::GfVec4h>();
		ret = { (float)vec[0], (float)vec[1], (float)vec[2] , (float)vec[3] };
	}
	return ret;
}

ZENO_HANDLE EvalUSDPrim::_makeTransformNode(ZENO_HANDLE main, const pxr::UsdGeomXformOp::Type& transType, const ZVARIANT& transVec) {
	ZENO_HANDLE newNode = Zeno_AddNode(main, "PrimitiveTransform"); 
	if (newNode == 0) {
		std::cout << "failed to create node named PrimitiveTransform" << std::endl;
		return 0;
	}

	if (transType == pxr::UsdGeomXformOp::TypeTranslate) {
		Zeno_SetInputDefl(main, newNode, "translation", std::get<zeno::vec3f>(transVec));
	} else if (transType == pxr::UsdGeomXformOp::TypeScale) {
		Zeno_SetInputDefl(main, newNode, "scaling", std::get<zeno::vec3f>(transVec));
	} else if (transType >= pxr::UsdGeomXformOp::TypeRotateX && transType <= pxr::UsdGeomXformOp::TypeRotateZ) {
		// only need to rotate one axis, so we don't care rotation order here
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateXYZ) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "XYZ");
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateXZY) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "XZY");
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateYXZ) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "YXZ");
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateYZX) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "YZX");
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateZXY) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "ZXY");
	} else if (transType == pxr::UsdGeomXformOp::TypeRotateZYX) {
		Zeno_SetInputDefl(main, newNode, "eulerXYZ", std::get<zeno::vec3f>(transVec));
		Zeno_SetParam(main, newNode, "EulerRotationOrder", "ZYX");
	} else if (transType == pxr::UsdGeomXformOp::TypeOrient) {
		Zeno_SetInputDefl(main, newNode, "quatRotation", std::get<zeno::vec4f>(transVec));
	} else if (transType == pxr::UsdGeomXformOp::TypeTransform) {
		; // ?? We don't support a single matrix node here, so...
	} else {
		Zeno_DeleteNode(main, newNode);
		return 0;
	}

	return newNode;
}
