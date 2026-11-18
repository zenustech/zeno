#ifndef __ROBOT_DYNAMIC_NODE_H__
#define __ROBOT_DYNAMIC_NODE_H__

#include "zenonode.h"

class RobotDynamicNodeUI : public ZenoNode
{
    Q_OBJECT
public:
    RobotDynamicNodeUI(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~RobotDynamicNodeUI();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onLoadInfoClicked();
};

#endif
