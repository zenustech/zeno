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

QGraphicsLayout* DynamicNumberNode::initParam(PARAM_CONTROL ctrl, const QString& name, const PARAM_INFO& param)
{
    return ZenoNode::initParam(ctrl, name, param);
}

QGraphicsLinearLayout* DynamicNumberNode::initCustomParamWidgets()
{
    return nullptr;

    //don't support editting legacy curve node.
    /*
    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem("curve", m_renderParams.paramFont, m_renderParams.paramClr.color());
    pHLayout->addItem(pNameItem);

    ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
    pHLayout->addItem(pEditBtn);
    connect(pEditBtn, SIGNAL(clicked()), this, SLOT(onEditClicked()));

    return pHLayout;
    */
}

void DynamicNumberNode::onEditClicked()
{
    ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);
    pEditor->show();
}