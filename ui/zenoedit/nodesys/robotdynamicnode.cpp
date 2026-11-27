#include "robotdynamicnode.h"
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/api.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include "zenomainwindow.h"
#include "timeline/ztimeline.h"

#include <fstream>
#include <string>
#include <iostream>

#include "tinygltf/json.hpp"

RobotDynamicNodeUI::RobotDynamicNodeUI(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{
}

RobotDynamicNodeUI::~RobotDynamicNodeUI()
{
}

ZGraphicsLayout* RobotDynamicNodeUI::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZGraphicsLayout* pHLayoutBtn = new ZGraphicsLayout(true);
    ZSimpleTextItem* pLabel = new ZSimpleTextItem("info");
    pLabel->setBrush(m_renderParams.socketClr.color());
    pLabel->setFont(m_renderParams.socketFont);
    pLabel->updateBoundingRect();
    pHLayoutBtn->addItem(pLabel);

    ZenoParamPushButton* pBtn = new ZenoParamPushButton("Load Info", -1, QSizePolicy::Expanding);
    pHLayoutBtn->addItem(pBtn);
    connect(pBtn, SIGNAL(clicked()), this, SLOT(onLoadInfoClicked()));

    _param_ctrl paramBtn;
    paramBtn.param_name = pLabel;
    paramBtn.param_control = pBtn;
    paramBtn.ctrl_layout = pHLayoutBtn;
    addParam(paramBtn);

    pHLayout->addLayout(pHLayoutBtn);
    return pHLayout;
}

void RobotDynamicNodeUI::onLoadInfoClicked()
{
    ZENO_HANDLE hGraph = Zeno_GetGraph("main");
    if (!hGraph)
        return;

    ZENO_HANDLE hNode = index().internalId();

    // 读取 dynamic_path 输入值
    ZVARIANT pathVar; std::string type;
    if (Zeno_GetInputDefl(hGraph, hNode, "dynamic_path", pathVar, type) != Err_NoError)
        return;
    if (!std::holds_alternative<std::string>(pathVar))
        return;
    std::string dynamicPath = std::get<std::string>(pathVar);
    if (dynamicPath.empty())
        return;

    // 解析动态 manifest JSON，统计帧数
    std::ifstream file(dynamicPath, std::ios::binary);
    if (!file.is_open())
        return;

    nlohmann::json manifest;
    try {
        file >> manifest;
    } catch (...) {
        return;
    }
    file.close();

    int nFrames = -1;
    if (manifest.contains("frames") && manifest["frames"].is_array()) {
        nFrames = static_cast<int>(manifest["frames"].size());
    } else if (manifest.contains("n_frames")) {
        nFrames = manifest["n_frames"].get<int>();
    }

    if (nFrames < 0)
        return;

    // 通过 API 更新 n_frames 参数（显示到节点上）
    Zeno_SetParam(hGraph, hNode, "n_frames", nFrames);

    // 设置时间轴帧范围（0 ~ nFrames-1）
    if (ZenoMainWindow* pMainWin = zenoApp->getMainWindow()) {
        if (ZTimeline* timeline = pMainWin->timeline()) {
            timeline->initFromTo(0, nFrames - 1);
        }
    }
}
