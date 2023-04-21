#include "cameranode.h"
#include "util/log.h"
#include <zenomodel/include/igraphsmodel.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/matrix_transform.hpp>

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

    // it seems no sense when we have multiple viewport but only one node.
    // which info of viewport will be synced to this node.
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
        zeno::scope_exit scope([=]() { pModel->endTransaction(); });

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

LightNode::LightNode(const NodeUtilParam &params, int pattern, QGraphicsItem *parent)
    : ZenoNode(params, parent)
{

}

LightNode::~LightNode() {

}

void LightNode::onEditClicked(){
    INPUT_SOCKETS inputs = index().data(ROLE_INPUTS).value<INPUT_SOCKETS>();

    const QString& nodeid = this->nodeId();
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget *> views = pWin->viewports();
    if (views.isEmpty())
        return;

    Zenovis* pZenovis = views[0]->getViewportWidget()->getZenoVis();
    ZASSERT_EXIT(pZenovis);

    auto sess = pZenovis->getSession();
    ZASSERT_EXIT(sess);
    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);

    PARAM_UPDATE_INFO info;

    std::vector<float> camProp = scene->getCameraProp();
    auto original_pos = glm::vec3(camProp[0], camProp[1], camProp[2]);
    auto pos = glm::normalize(glm::vec3(camProp[0], camProp[1], camProp[2]));
    auto view = -1.0f * glm::normalize(glm::vec3(camProp[3], camProp[4], camProp[5]));
    auto up = glm::normalize(glm::vec3(camProp[6], camProp[7], camProp[8]));
    auto right = glm::normalize(glm::cross(up, view));

    glm::mat3 rotation(right, up, view);

    glm::mat4 correct_matrix = glm::mat4(1.0f);
    correct_matrix = glm::rotate(correct_matrix, glm::radians(90.0f), right);
    glm::mat3x3 corrected_matrix = glm::mat3x3(correct_matrix);
    glm::mat3 matC = corrected_matrix * rotation;

    glm::quat quat = glm::quat_cast(matC);

    INPUT_SOCKET position = inputs["position"];
    UI_VECTYPE vec({ original_pos[0], original_pos[1], original_pos[2] });
    info.name = "position";
    info.oldValue = position.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);

    INPUT_SOCKET rotate = inputs["quaternion"];
    vec = {quat.w, quat.x, quat.y, quat.z};
    info.name = "quaternion";
    info.oldValue = rotate.info.defaultValue;
    info.newValue = QVariant::fromValue(vec);
    pModel->updateSocketDefl(nodeid, info, this->subgIndex(), true);
}

ZGraphicsLayout* LightNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem *pNameItem = new ZSimpleTextItem("sync");
    pNameItem->setBrush(m_renderParams.socketClr.color());
    pNameItem->setFont(m_renderParams.socketFont);
    pNameItem->updateBoundingRect();

    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
}
