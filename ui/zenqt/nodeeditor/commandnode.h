#ifndef __COMMAND_NODE_H__
#define __COMMAND_NODE_H__

#include "zenonode.h"

class CommandNode : public ZenoNode
{
    Q_OBJECT
public:
    CommandNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~CommandNode();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;
    Callback_OnButtonClicked registerButtonCallback(const QModelIndex& paramIdx) override;

private slots:
    void onExecuteClicked();
    void onGenerateClicked();
};

#endif