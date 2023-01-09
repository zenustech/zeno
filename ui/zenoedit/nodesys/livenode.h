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
    QGraphicsLinearLayout* initCustomParamWidgets() override;

  private slots:
    void onSyncClicked();
};

class LiveCameraNode : public ZenoNode
{
    Q_OBJECT
  public:
    LiveCameraNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~LiveCameraNode();

  protected:
    QGraphicsLinearLayout* initCustomParamWidgets() override;

  private slots:
    void onSyncClicked();
};

#endif //ZENO_LIVENODE_H
