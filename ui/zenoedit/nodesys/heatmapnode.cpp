#include "heatmapnode.h"
#include <zenoui/comctrl/dialog/zenoheatmapeditor.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "util/log.h"
#include "util/apphelper.h"
#include <zenomodel/include/uihelper.h>


MakeHeatMapNode::MakeHeatMapNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

MakeHeatMapNode::~MakeHeatMapNode()
{

}

ZGraphicsLayout* MakeHeatMapNode::initParam(PARAM_CONTROL ctrl, const QString& name, const PARAM_INFO& param, ZenoSubGraphScene* pScene)
{
    if (param.control == CONTROL_COLOR)
    {
        ZASSERT_EXIT(name == "_RAMPS", nullptr);

        ZGraphicsLayout* pParamLayout = new ZGraphicsLayout(true);
        ZSimpleTextItem* pNameItem = new ZSimpleTextItem("color");
        pNameItem->setBrush(m_renderParams.socketClr.color());
        pNameItem->setFont(m_renderParams.socketFont);
        pNameItem->updateBoundingRect();

        pParamLayout->addItem(pNameItem);

        ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
        pParamLayout->addItem(pEditBtn);
        connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));
        return pParamLayout;
    }
    else
    {
        return ZenoNode::initParam(ctrl, name, param, pScene);
    }
}

void MakeHeatMapNode::onEditClicked()
{
    PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    if (params.find("color") != params.end())
    {
        PARAM_UPDATE_INFO info;
        PARAM_INFO& param = params["color"];
        info.name = "color";
        info.oldValue = param.value;
        QLinearGradient grad = param.value.value<QLinearGradient>();

        ZenoHeatMapEditor editor(grad);
        editor.exec();
        QLinearGradient newGrad = editor.colorRamps();
        if (newGrad != grad)
        {
            info.newValue = QVariant::fromValue(newGrad);
            IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
            pModel->updateParamInfo(nodeId(), info, subGraphIndex(), true);
        }
    }
    else if (params.find("_RAMPS") != params.end())
    {
        //legacy format
        PARAM_INFO& param = params["_RAMPS"];
        const QString& oldColor = param.value.toString();
        QLinearGradient grad = AppHelper::colorString2Grad(oldColor);

        ZenoHeatMapEditor editor(grad);
        editor.exec();

        QLinearGradient newGrad = editor.colorRamps();
        QString colorText = UiHelper::gradient2colorString(newGrad);
        if (colorText != oldColor)
        {
            PARAM_UPDATE_INFO info;
            info.name = "_RAMPS";
            info.oldValue = oldColor;
            info.newValue = colorText;
            IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
            pModel->updateParamInfo(nodeId(), info, subGraphIndex(), true);
        }
    }
}