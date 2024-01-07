#ifndef __HEATMAP_NODE_H__
#define __HEATMAP_NODE_H__

#include "zenonode.h"

class MakeHeatMapNode : public ZenoNode
{
	Q_OBJECT
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