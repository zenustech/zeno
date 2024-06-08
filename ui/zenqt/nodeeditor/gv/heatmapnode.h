#ifndef __HEATMAP_NODE_H__
#define __HEATMAP_NODE_H__

#include "zenonodenew.h"

class MakeHeatMapNode : public ZenoNodeNew
{
    Q_OBJECT
    typedef ZenoNodeNew _base;
public:
	MakeHeatMapNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~MakeHeatMapNode();

protected:
	ZGraphicsLayout* initCustomParamWidgets() override;

private slots:
	void onEditClicked();

private:

};

#endif