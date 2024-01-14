#include "heatmapnode.h"
#include "dialog/zenoheatmapeditor.h"
#include "zenoapplication.h"
#include "model/graphsmanager.h"
#include "util/log.h"
#include "util/apphelper.h"
#include "util/uihelper.h"


MakeHeatMapNode::MakeHeatMapNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

MakeHeatMapNode::~MakeHeatMapNode()
{

}

ZGraphicsLayout* MakeHeatMapNode::initCustomParamWidgets()
{
    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZSimpleTextItem* pNameItem = new ZSimpleTextItem("color");
    pNameItem->setBrush(m_renderParams.socketClr.color());
    pNameItem->setFont(m_renderParams.socketFont);
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn, Qt::AlignRight);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    _param_ctrl param;
    param.param_name = pNameItem;
    param.param_control = pEditBtn;
    param.ctrl_layout = pHLayout;
    addParam(param);

    return pHLayout;
}

void MakeHeatMapNode::onEditClicked()
{
    QPersistentModelIndex nodeIdx = index();
    PARAMS_INFO params = nodeIdx.data(ROLE_INPUTS).value<PARAMS_INFO>();
    if (params.find("color") != params.end())
    {
        zeno::ParamInfo& param = params["color"];
        param.defl;
        //TODO: convert defl to QLinearGradient
        QLinearGradient grad;

        ZenoHeatMapEditor editor(grad);
        editor.exec();
        QLinearGradient newGrad = editor.colorRamps();
        if (newGrad != grad)
        {
            //TODO: convert QLinearGradient to defl.
            //param.defl = ...
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
            pModel->setData(nodeIdx, QVariant::fromValue(params), ROLE_INPUTS);
        }
    }
    else if (params.find("_RAMPS") != params.end())
    {
        //deprecated
        /*
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
            IGraphsModel *pModel = zenoApp->graphsManager()->currentModel();
            pModel->updateParamInfo(nodeId(), info, subGraphIndex(), true);
        }
        */
    }
}