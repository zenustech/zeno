#ifndef __CAMERA_NODE_H__
#define __CAMERA_NODE_H__

#include "zenonode.h"
#include "zeno/utils/vec.h"

class CameraNode : public ZenoNode
{
    Q_OBJECT
public:
    CameraNode(const NodeUtilParam& params, int pattern = 0, QGraphicsItem* parent = nullptr);
    ~CameraNode();

    int CameraPattern = 0;

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};

class LightNode : public ZenoNode
{
    Q_OBJECT
public:
    LightNode(const NodeUtilParam& params, int pattern = 0, QGraphicsItem* parent = nullptr);
    ~LightNode();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};

class PrimitiveTransform : public ZenoNode
{
Q_OBJECT
public:
    PrimitiveTransform(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~PrimitiveTransform();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void movePivotToCentroid();
    void moveCentroidToOrigin();
};

#endif