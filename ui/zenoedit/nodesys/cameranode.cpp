#include "cameranode.h"
#include "util/log.h"
#include <zenoui/include/igraphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"

#include <viewport/zenovis.h>
#include "zenovis/Session.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>

// FIXME  fatal error C1189: #error:  OpenGL header already included,
//  remove this include, glad already provides it
//#include "zenovis/Camera.h"

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

    auto &inst = Zenovis::GetInstance();
    auto sess = inst.getSession();
    ZASSERT_EXIT(sess);

    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);

    UI_VECTYPE vec({ 0., 0., 0. });

    PARAM_UPDATE_INFO info;

    pModel->beginTransaction("update camera info");

    INPUT_SOCKET pos = inputs["pos"];
    //vec = {scene->camera->m_lodcenter.x, scene->camera->m_lodcenter.y, scene->camera->m_lodcenter.z};
    std::vector<float> camProp = scene->getCameraProp();
    vec = {camProp[0], camProp[1], camProp[2]};
    info.name = "pos";
    info.oldValue = pos.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET up = inputs["up"];
    vec = {camProp[6], camProp[7], camProp[8]};
    info.name = "up";
    info.oldValue = up.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET view = inputs["view"];
    vec = {camProp[3], camProp[4], camProp[5]};
    info.name = "view";
    info.oldValue = view.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET frame = inputs["frame"];
    // FIXME Not work
    int frameId = sess->get_curr_frameid();
    frameId = zeno::getSession().globalState->frameid;

    info.name = "frame";
    info.oldValue = frame.info.defaultValue;
    frame.info.defaultValue = QVariant::fromValue(frameId);
}