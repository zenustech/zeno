#ifndef __MAKE_DICT_NODE_H__
#define __MAKE_DICT_NODE_H__

#include "zenonode.h"

class MakeDictNode : public ZenoNode
{
    Q_OBJECT
public:
    MakeDictNode(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~MakeDictNode();

protected:
    QGraphicsLayout* initParams();
    QGraphicsLinearLayout* initCustomParamWidgets() override;
};

#endif