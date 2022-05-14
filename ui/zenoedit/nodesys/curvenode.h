#ifndef __CURVE_NODE_H__
#define __CURVE_NODE_H__

#include "zenonode.h"

class MakeCurveNode : public ZenoNode
{
	Q_OBJECT
public:
	MakeCurveNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~MakeCurveNode();

protected:
	QGraphicsLayout* initParams() override;
	void initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& name, const PARAM_INFO& param) override;
	QGraphicsLinearLayout* initCustomParamWidgets() override;

private slots:
	void onEditClicked();

private:

};

#endif
