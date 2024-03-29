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
	std::cout << "you clicked eval button!" << std::endl;
	
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

	if (primType == "Sphere") {
		_emitCreateSphereNode(mainGraph);
	} else if (primType == "Capsule"){
		_emitCreateCapsuleNode(mainGraph);
	} else if (primType == "Cube"){
		_emitCreateCubeNode(mainGraph);
	} else if (primType == "Cylinder"){
		_emitCreateCylinderNode(mainGraph);
	} else if (primType == "Cone"){
		_emitCreateConeNode(mainGraph);
	} else if (primType == "Plane"){
		_emitCreatePlaneNode(mainGraph);
	} else if (primType == "Mesh") {
		;
	}
}

void EvalUSDPrim::_emitCreateSphereNode(ZENO_HANDLE targetGraph) {
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateSphere");
	if (newNode == 0) {
		std::cout << "failed to emit CreateSphere node" << std::endl;
		return;
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
}

// TODO: we need a CreateCapsule node
void EvalUSDPrim::_emitCreateCapsuleNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCapsule");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCapsule node" << std::endl;
		return;
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
}

void EvalUSDPrim::_emitCreateCubeNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCube");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCube node" << std::endl;
		return;
	}

	ZENO_HANDLE curNode = index().internalId();
	std::pair<float, float> nodePos;
	Zeno_GetPos(targetGraph, curNode, nodePos);

	double size;
	mUSDPrim.GetAttribute(pxr::TfToken("size")).Get(&size);
	Zeno_SetInputDefl(targetGraph, newNode, "size", size);
	Zeno_SetPos(targetGraph, newNode, nodePos);
	Zeno_SetView(targetGraph, newNode, true);
}

void EvalUSDPrim::_emitCreateCylinderNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCylinder");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCylinder node" << std::endl;
		return;
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
}

void EvalUSDPrim::_emitCreateConeNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreateCone");
	if (newNode == 0) {
		std::cout << "failed to emit CreateCone node" << std::endl;
		return;
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
}

void EvalUSDPrim::_emitCreatePlaneNode(ZENO_HANDLE targetGraph){
	ZENO_HANDLE newNode = Zeno_AddNode(targetGraph, "CreatePlane");
	if (newNode == 0) {
		std::cout << "failed to emit CreatePlane node" << std::endl;
		return;
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
}

void EvalUSDPrim::_emitPrimitiveTransformNode(ZENO_HANDLE targetGraph) {
	auto xform = pxr::UsdGeomXform(mUSDPrim);
	bool resetXformStack;
	auto xformOps = xform.GetOrderedXformOps(&resetXformStack);

	// TODO: traverse and parse xform ops into PrimitiveTransform nodes
	;
}
