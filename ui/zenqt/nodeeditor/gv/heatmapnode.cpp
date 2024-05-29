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

    //_param_ctrl param;
    //param.param_name = pNameItem;
    //param.param_control = pEditBtn;
    //param.ctrl_layout = pHLayout;
    //addParam(param);

    return pHLayout;
}

void MakeHeatMapNode::onEditClicked()
{
    QPersistentModelIndex nodeIdx = index();
    PARAMS_INFO params = nodeIdx.data(ROLE_INPUTS).value<PARAMS_INFO>();
        QString heatmap = nodeIdx.data(ROLE_PARAM_VALUE).toString();

        ZenoHeatMapEditor editor(heatmap);
        editor.exec();
        QString newVal = editor.colorRamps();
        if (newVal != heatmap)
        {
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(nodeIdx.model());
            pModel->setData(nodeIdx, QVariant::fromValue(newVal), ROLE_PARAM_VALUE);
        }
}