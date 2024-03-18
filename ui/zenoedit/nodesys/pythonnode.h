#ifndef __PYTHON_NODE_H__
#define __PYTHON_NODE_H__

#include "zenonode.h"

class PythonNode : public ZenoNode
{
    Q_OBJECT
public:
    PythonNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~PythonNode();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onExecuteClicked();
    void onGenerateClicked();
};

#endif