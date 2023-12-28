#ifndef __PYTHON_MATERIAL_NODE_H__
#define __PYTHON_MATERIAL_NODE_H__

#include "zenonode.h"

class PythonMaterialNode : public ZenoNode
{
    Q_OBJECT
public:
    PythonMaterialNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~PythonMaterialNode();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onExecuteClicked();
    void onEditClicked();
private:
    QString m_jsonStr;
};

#endif