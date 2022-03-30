#ifndef __CURVEMAP_NODE_H__
#define __CURVEMAP_NODE_H__

#include "zenonode.h"

class MakeCurvemapNode : public ZenoNode
{
	Q_OBJECT
public:
	MakeCurvemapNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~MakeCurvemapNode();

protected:
	QGraphicsLayout* initParams() override;
	void initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param) override;
	QGraphicsLinearLayout* initCustomParamWidgets() override;

private slots:
	void onEditClicked();

private:

};

#endif