#ifndef __SUBNETNODE_H__
#define __SUBNETNODE_H__

#include "zenonode.h"

class SubnetNode : public ZenoNode
{
	Q_OBJECT
	typedef ZenoNode _base;
public:
	SubnetNode(bool bInput, const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~SubnetNode();

protected:
	void onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QVariant& textValue) override;

private:
	bool m_bInput;
};

#endif