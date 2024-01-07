#include "launch/livehttpserver.h"
#include "launch/livetcpserver.h"
#include "livenode.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/api.h>
#include <zenomodel/include/graphsmanagment.h>

#include <chrono>

LiveMeshNode::LiveMeshNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
      connect(zenoApp->getMainWindow()->liveSignalsBridge, &LiveSignalsBridge::frameMeshSendDone, this, [&](){
                  std::cout << "sendDone frame mesh\n";

                  onSyncClicked();
              });
}

LiveMeshNode::~LiveMeshNode()
{

}

ZGraphicsLayout *LiveMeshNode::initCustomParamWidgets() {
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem *pNameItem = new ZSimpleTextItem("node");
    pNameItem->setBrush(m_renderParams.socketClr.color());
    pNameItem->setFont(m_renderParams.socketFont);
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pSyncBtn = new ZenoParamPushButton("Sync", -1, QSizePolicy::Expanding);
    ZenoParamPushButton* pCleanBtn = new ZenoParamPushButton("Clean", -1, QSizePolicy::Expanding);
    QGraphicsWidget *widget = new QGraphicsWidget;
    QGraphicsLinearLayout *layout = new QGraphicsLinearLayout(widget);
    layout->setContentsMargins(20, 0, 0, 0);
    widget->setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    layout->addItem(pSyncBtn);
    layout->addItem(pCleanBtn);
    pHLayout->addItem(widget);
    connect(pSyncBtn, SIGNAL(clicked()), this, SLOT(onSyncClicked()));
    connect(pCleanBtn, SIGNAL(clicked()), this, SLOT(onCleanClicked()));
    
    _param_ctrl param;
    param.param_name = pNameItem;
    param.param_control = widget;
    param.ctrl_layout = pHLayout;
    addParam(param);

    return pHLayout;
}

void LiveMeshNode::onSyncClicked() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    auto liveData = to_string(zenoApp->getMainWindow()->liveHttpServer->d_frame_mesh);
    ZENO_HANDLE hSubg = subgIndex().internalId();
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(hSubg, liveNode, "vertSrc", std::move(liveData));

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Used time onSyncClicked " << ms_int.count() << " ms\n";
    std::cout << "Used time onSyncClicked " << ms_double.count() << " ms\n";
}

void LiveMeshNode::onCleanClicked() {
    ZENO_HANDLE hSubg = subgIndex().internalId();
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(hSubg, liveNode, "vertSrc", std::string("{}"));
    zenoApp->getMainWindow()->liveHttpServer->d_frame_mesh.clear();
}
