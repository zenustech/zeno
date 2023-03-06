#ifndef __CAMERA_NODE_H__
#define __CAMERA_NODE_H__

#include "zenonode.h"

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


#endif