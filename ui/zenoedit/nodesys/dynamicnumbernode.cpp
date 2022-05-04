#include "dynamicnumbernode.h"
#include "../curvemap/zcurvemapeditor.h"


DynamicNumberNode::DynamicNumberNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent)
{

}

DynamicNumberNode::~DynamicNumberNode()
{

}

QGraphicsLayout* DynamicNumberNode::initParams()
{
    return ZenoNode::initParams();
}

void DynamicNumberNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout *pParamLayout, const QString &name, const PARAM_INFO &param)
{
    ZenoNode::initParam(ctrl, pParamLayout, name, param);
}

QGraphicsLinearLayout* DynamicNumberNode::initCustomParamWidgets()
{
    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem("curve", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
}

void DynamicNumberNode::onEditClicked()
{
    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);
    pEditor->show();
}