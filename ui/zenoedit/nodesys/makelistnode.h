#ifndef __MAKELIST_NODE_H__
#define __MAKELIST_NODE_H__

#include "zenonode.h"

class MakeListNode : public ZenoNode
{
    Q_OBJECT
public:
    MakeListNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~MakeListNode();

protected:
    QGraphicsLayout* initParams() override;
    QGraphicsLayout* initParam(PARAM_CONTROL ctrl, const QString& name, const PARAM_INFO& param) override;
    QGraphicsLinearLayout* initCustomParamWidgets() override;
};


#endif