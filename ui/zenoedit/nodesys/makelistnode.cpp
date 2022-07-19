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

QGraphicsLayout* MakeListNode::initParam(PARAM_CONTROL ctrl, const QString& name, const PARAM_INFO& param)
{
    return ZenoNode::initParam(ctrl, name, param);
}

QGraphicsLinearLayout* MakeListNode::initCustomParamWidgets()
{
    return ZenoNode::initCustomParamWidgets();
}
