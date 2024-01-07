#ifndef ZENO_LIVENODE_H
#define ZENO_LIVENODE_H

#include "zenonode.h"

class LiveMeshNode : public ZenoNode
{
    Q_OBJECT
  public:
    LiveMeshNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~LiveMeshNode();

  protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

  public slots:
    void onSyncClicked();
    void onCleanClicked();
};

#endif //ZENO_LIVENODE_H
