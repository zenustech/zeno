#ifndef __SUBNETNODE_H__
#define __SUBNETNODE_H__

#include "zenonode.h"

class SubnetNode : public ZenoNode
{
    Q_OBJECT
    typedef ZenoNode _base;
public:
    SubnetNode(bool bInput, const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~SubnetNode();

private:
    bool m_bInput;
};

#endif