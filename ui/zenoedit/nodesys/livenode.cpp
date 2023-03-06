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

    ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem("node", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pSyncBtn = new ZenoParamPushButton("Sync", -1, QSizePolicy::Expanding);
    ZenoParamPushButton* pCleanBtn = new ZenoParamPushButton("Clean", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pSyncBtn);
    pHLayout->addItem(pCleanBtn);
    connect(pSyncBtn, SIGNAL(clicked()), this, SLOT(onSyncClicked()));
    connect(pCleanBtn, SIGNAL(clicked()), this, SLOT(onCleanClicked()));

    return pHLayout;
}

void LiveMeshNode::onSyncClicked() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    auto liveData = to_string(zenoApp->getMainWindow()->liveHttpServer->d_frame_mesh);
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(liveNode, "vertSrc", std::move(liveData));

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Used time onSyncClicked " << ms_int.count() << " ms\n";
    std::cout << "Used time onSyncClicked " << ms_double.count() << " ms\n";
}

void LiveMeshNode::onCleanClicked() {
    ZENO_HANDLE liveNode = index().internalId();
    Zeno_SetInputDefl(liveNode, "vertSrc", std::string("{}"));
    zenoApp->getMainWindow()->liveHttpServer->d_frame_mesh.clear();
}
