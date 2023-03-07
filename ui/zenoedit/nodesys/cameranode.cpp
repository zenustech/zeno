#include "cameranode.h"
#include "util/log.h"
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <viewport/zenovis.h>
#include "zenovis/Session.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalState.h>

CameraNode::CameraNode(const NodeUtilParam& params, int pattern, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
    CameraPattern = pattern;
}

CameraNode::~CameraNode()
{

}

ZGraphicsLayout* CameraNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem* pNameItem = new ZSimpleTextItem("sync");
    pNameItem->setBrush(m_renderParams.socketClr.color());
    pNameItem->setFont(m_renderParams.socketFont);
    pNameItem->updateBoundingRect();

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
        inputs.find("view") != inputs.end());

    const QString& nodeid = this->nodeId();
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget*> views = pWin->viewports();
    for (auto pDisplay : views)
    {
        ZASSERT_EXIT(pDisplay);
        ViewportWidget* pViewport = pDisplay->getViewportWidget();
        ZASSERT_EXIT(pViewport);
        auto sess = pViewport->getSession();
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

        INPUT_SOCKET fov = inputs["fov"];
        info.name = "fov";
        info.oldValue = fov.info.defaultValue;
        info.newValue = QVariant::fromValue(camProp[9]);
        pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

        INPUT_SOCKET aperture = inputs["aperture"];
        info.name = "aperture";
        info.oldValue = aperture.info.defaultValue;
        info.newValue = QVariant::fromValue(camProp[10]);
        pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

        INPUT_SOCKET focalPlaneDistance = inputs["focalPlaneDistance"];
        info.name = "focalPlaneDistance";
        info.oldValue = focalPlaneDistance.info.defaultValue;
        info.newValue = QVariant::fromValue(camProp[11]);
        pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

        // Is CameraNode
        if(CameraPattern == 0) {
            INPUT_SOCKET frame = inputs["frame"];
            // FIXME Not work
            int frameId = sess->get_curr_frameid();
            frameId = zeno::getSession().globalState->frameid;

            info.name = "frame";
            info.oldValue = frame.info.defaultValue;
            frame.info.defaultValue = QVariant::fromValue(frameId);

            INPUT_SOCKET other = inputs["other"];
            std::string other_prop;
            for (int i = 12; i < camProp.size(); i++)
                other_prop += std::to_string(camProp[i]) + ",";
            info.name = "other";
            info.oldValue = other.info.defaultValue;
            info.newValue = QVariant::fromValue(QString(other_prop.c_str()));
            pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);
        }

        // Is MakeCamera
        if(CameraPattern == 1){
            // Here
        }
    }
}