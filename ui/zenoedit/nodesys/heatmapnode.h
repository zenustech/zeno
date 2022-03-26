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
	QGraphicsLayout* initParams() override;
	void initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param) override;
	//QGraphicsLinearLayout* initCustomParamWidgets() override;

private slots:
	void onEditClicked();

private:

};


#endif