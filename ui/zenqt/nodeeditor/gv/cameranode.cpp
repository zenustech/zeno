#include "cameranode.h"
#include "util/log.h"
#include "control/renderparam.h"
#include "zenoapplication.h"
#include "model/graphstreemodel.h"
#include "zenomainwindow.h"
#include "viewport/viewportwidget.h"
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "viewport/displaywidget.h"
#include <viewport/zenovis.h>
#include "zenovis/Session.h"
#include "zenovis/Camera.h"
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

    //_param_ctrl param;
    //param.param_name = pNameItem;
    //param.param_control = pEditBtn;
    //param.ctrl_layout = pHLayout;
    //addParam(param);

    return pHLayout;
}

void CameraNode::onEditClicked()
{
    zeno::ParamPrimitive param;

    QPersistentModelIndex nodeIdx = index();
    ZASSERT_EXIT(nodeIdx.isValid());
    QAbstractItemModel* pGraphM = const_cast<QAbstractItemModel*>(nodeIdx.model());
    ZASSERT_EXIT(pGraphM);

    PARAMS_INFO inputs = nodeIdx.data(ROLE_INPUTS).value<PARAMS_INFO>();
    ZASSERT_EXIT(inputs.find("pos") != inputs.end() &&
        inputs.find("up") != inputs.end() &&
        inputs.find("view") != inputs.end());

    const QString& nodeid = this->nodeId();

    //IGraphsModel* pModel = zenoApp->graphsManager()->currentModel();
    //ZASSERT_EXIT(pModel);

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    // it seems no sense when we have multiple viewport but only one node.
    // which info of viewport will be synced to this node.
    DisplayWidget* pDisplay = pWin->getCurrentViewport();
    if (pDisplay)
    {
        auto pZenoVis = pDisplay->getZenoVis();
        ZASSERT_EXIT(pZenoVis);
        auto sess = pZenoVis->getSession();
        ZASSERT_EXIT(sess);

        auto scene = sess->get_scene();
        ZASSERT_EXIT(scene);

        zeno::vec3f vec = { 0., 0., 0. };
       

        auto camera = *(scene->camera.get());

        zeno::ParamPrimitive& pos = inputs["pos"];
        vec = {camera.m_lodcenter[0], camera.m_lodcenter[1], camera.m_lodcenter[2]};
        pos.defl = vec;

        inputs["up"].defl = zeno::vec3f({ camera.m_lodup[0], camera.m_lodup[1], camera.m_lodup[2] });
        inputs["view"].defl = zeno::vec3f({ camera.m_lodfront[0], camera.m_lodfront[1], camera.m_lodfront[2] });
        inputs["fov"].defl = camera.m_fov;
        inputs["aperture"].defl = camera.m_aperture;
        inputs["focalPlaneDistance"].defl = camera.focalPlaneDistance;

        // Is CameraNode
        if(CameraPattern == 0) {
            // FIXME Not work
            int frameId = sess->get_curr_frameid();
            frameId = zeno::getSession().globalState->getFrameId();
            inputs["frame"].defl = frameId;

            std::string other_prop;
            auto center = camera.m_center;
            other_prop += zeno::format("{},{},{},", center[0], center[1], center[2]);
            other_prop += zeno::format("{},", camera.m_theta);
            other_prop += zeno::format("{},", camera.m_phi);
            other_prop += zeno::format("{},", camera.m_radius);

            inputs["other"].defl = other_prop;
        }

        // Is MakeCamera
        if(CameraPattern == 1){
            // Here
        }

        pGraphM->setData(nodeIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
	}
}

LightNode::LightNode(const NodeUtilParam &params, int pattern, QGraphicsItem *parent)
    : ZenoNode(params, parent)
{

}

LightNode::~LightNode() {

}

void LightNode::onEditClicked()
{
    QPersistentModelIndex nodeIdx = index();
    ZASSERT_EXIT(nodeIdx.isValid());
    QAbstractItemModel* pGraphM = const_cast<QAbstractItemModel*>(nodeIdx.model());
    ZASSERT_EXIT(pGraphM);

    PARAMS_INFO inputs = nodeIdx.data(ROLE_INPUTS).value<PARAMS_INFO>();

    const QString& nodeid = this->nodeId();

    ZenoMainWindow *pWin = zenoApp->getMainWindow();
    ZASSERT_EXIT(pWin);

    QVector<DisplayWidget*> views = pWin->viewports();
    if (views.isEmpty())
        return;

    //todo: case about camera on multiple viewports.
    auto pZenoVis = views[0]->getZenoVis();
    ZASSERT_EXIT(pZenoVis);

    auto sess = pZenoVis->getSession();
    ZASSERT_EXIT(sess);
    auto scene = sess->get_scene();
    ZASSERT_EXIT(scene);

    auto camera = *(scene->camera.get());
    auto original_pos = glm::vec3(camera.m_lodcenter);
//    auto pos = glm::normalize(glm::vec3(camProp[0], camProp[1], camProp[2]));
    auto view = -1.0f * glm::normalize(camera.m_lodfront);
    auto up = glm::normalize(camera.m_lodup);
    auto right = glm::normalize(glm::cross(up, view));

    glm::mat3 rotation(right, up, view);

    glm::mat4 correct_matrix = glm::mat4(1.0f);
    correct_matrix = glm::rotate(correct_matrix, glm::radians(90.0f), right);
    glm::mat3x3 corrected_matrix = glm::mat3x3(correct_matrix);
    glm::mat3 matC = corrected_matrix * rotation;

    glm::quat quat = glm::quat_cast(matC);

    inputs["position"].defl = zeno::vec3f({ original_pos[0], original_pos[1], original_pos[2] });
    inputs["quaternion"].defl = zeno::vec3f({ quat.w, quat.x, quat.y, quat.z });
    pGraphM->setData(nodeIdx, QVariant::fromValue(inputs), ROLE_INPUTS);
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

    //_param_ctrl param;
    //param.param_name = pNameItem;
    //param.param_control = pEditBtn;
    //param.ctrl_layout = pHLayout;
    //addParam(param);

    return pHLayout;
}
