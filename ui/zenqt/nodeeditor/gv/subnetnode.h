#ifndef __SUBNETNODE_H__
#define __SUBNETNODE_H__

#include "zenonodenew.h"

class SubnetNode : public ZenoNodeNew
{
    Q_OBJECT
    typedef ZenoNodeNew _base;
public:
    SubnetNode(bool bInput, const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~SubnetNode();

private:
    bool m_bInput;
};

#endif