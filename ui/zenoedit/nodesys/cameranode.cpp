#include "cameranode.h"
#include "util/log.h"
#include <zenoui/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"


CameraNode::CameraNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

CameraNode::~CameraNode()
{

}

QGraphicsLinearLayout* CameraNode::initCustomParamWidgets()
{
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("sync", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
}

void CameraNode::onEditClicked()
{
    INPUT_SOCKETS inputs = index().data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    ZASSERT_EXIT(inputs.find("pos") != inputs.end() &&
        inputs.find("up") != inputs.end() &&
        inputs.find("view") != inputs.end() &&
        inputs.find("frame") != inputs.end());

    const QString& nodeid = this->nodeId();
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    UI_VECTYPE vec({ 0., 0., 0. });

    PARAM_UPDATE_INFO info;

    pModel->beginTransaction("update camera info");

    INPUT_SOCKET pos = inputs["pos"];
    //TODO: get pos from viewport camera.
    //vec = ...
    info.name = "pos";
    info.oldValue = pos.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET up = inputs["up"];
    //TODO: get pos from viewport camera.
    //vec = ...
    info.name = "up";
    info.oldValue = up.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET view = inputs["view"];
    //TODO: get view from viewport camera.
    //vec = ...
    info.name = "view";
    info.oldValue = view.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET frame = inputs["frame"];
    //TODO: get frame from viewport camera.
    //frame = ...
    info.name = "frame";
    info.oldValue = frame.info.defaultValue;
    frame.info.defaultValue = QVariant::fromValue(vec);
}