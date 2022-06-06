#include "makelistnode.h"


MakeListNode::MakeListNode(const NodeUtilParam& params, QGraphicsItem* parent)
    : ZenoNode(params, parent)
{

}

MakeListNode::~MakeListNode()
{

}

QGraphicsLayout* MakeListNode::initParams()
{
    return ZenoNode::initParams();
}

void MakeListNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param)
{
    ZenoNode::initParam(ctrl, pParamLayout, name, param);
}

QGraphicsLinearLayout* MakeListNode::initCustomParamWidgets()
{
    return ZenoNode::initCustomParamWidgets();
}
