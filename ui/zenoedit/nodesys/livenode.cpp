#include "launch/livetcpserver.h"
#include "livenode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/api.h>
#include <zenomodel/include/graphsmanagment.h>

LiveMeshNode::LiveMeshNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
      connect(zenoApp->getMainWindow()->liveTcpServer, &LiveTcpServer::sendVertDone, this, [&](){
                  std::cout << "sendDone vert slot\n";

                  onSyncClicked();
              });
}

LiveMeshNode::~LiveMeshNode()
{

}

QGraphicsLinearLayout *LiveMeshNode::initCustomParamWidgets() {
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("node", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pSyncBtn = new ZenoParamPushButton("Sync", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pSyncBtn);
    connect(pSyncBtn, SIGNAL(clicked()), this, SLOT(onSyncClicked()));

    return pHLayout;
}

void LiveMeshNode::onSyncClicked() {
    auto liveData = zenoApp->getMainWindow()->liveTcpServer->liveData;
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(liveNode, "vertSrc", liveData.verSrc);
}

LiveCameraNode::LiveCameraNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
    connect(zenoApp->getMainWindow()->liveTcpServer, &LiveTcpServer::sendCamDone, this, [&](){
        std::cout << "sendDone came slot\n";

        onSyncClicked();
    });
}

LiveCameraNode::~LiveCameraNode()
{

}

QGraphicsLinearLayout *LiveCameraNode::initCustomParamWidgets() {
    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("node", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pSyncBtn = new ZenoParamPushButton("Sync", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pSyncBtn);
    connect(pSyncBtn, SIGNAL(clicked()), this, SLOT(onSyncClicked()));

    return pHLayout;
}

void LiveCameraNode::onSyncClicked() {
    auto liveData = zenoApp->getMainWindow()->liveTcpServer->liveData;
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(liveNode, "camSrc", liveData.camSrc);
}