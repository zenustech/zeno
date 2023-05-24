#ifndef __READ_FBX_PRIM_H__
#define __READ_FBX_PRIM_H__

#include "zenonode.h"

class ReadFBXPrim : public ZenoNode
{
    Q_OBJECT
public:
    ReadFBXPrim(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~ReadFBXPrim();

protected:
    ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
    void onEditClicked();
};


#endif