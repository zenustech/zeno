#ifndef __CAMERA_NODE_H__
#define __CAMERA_NODE_H__

#include "zenonodenew.h"

class CameraNode : public ZenoNodeNew
{
    Q_OBJECT
    typedef ZenoNodeNew _base;
public:
    CameraNode(const NodeUtilParam& params, int pattern = 0, QGraphicsItem* parent = nullptr);
    ~CameraNode();

    int CameraPattern = 0;

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};

class LightNode : public ZenoNodeNew
{
    Q_OBJECT
    typedef ZenoNodeNew _base;
public:
    LightNode(const NodeUtilParam& params, int pattern = 0, QGraphicsItem* parent = nullptr);
    ~LightNode();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};


#endif