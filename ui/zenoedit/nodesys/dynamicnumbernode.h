#ifndef __DYNAMIC_NUMBER_NODE_H__
#define __DYNAMIC_NUMBER_NODE_H__

#include "zenonode.h"

class DynamicNumberNode : public ZenoNode
{
    Q_OBJECT
public:
    DynamicNumberNode(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~DynamicNumberNode();

protected:
    QGraphicsLayout *initParams() override;
    void initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout *pParamLayout,
                   const QString &name, const PARAM_INFO &param) override;
    QGraphicsLinearLayout *initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};


#endif