#ifndef __SUBNETNODE_H__
#define __SUBNETNODE_H__

#include "zenonode.h"

class SubInputNode : public ZenoNode
{
	Q_OBJECT
	typedef ZenoNode _base;
public:
	SubInputNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~SubInputNode();

protected:
	void onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue) override;
};

class SubOutputNode : public ZenoNode
{
	Q_OBJECT
	typedef ZenoNode _base;
public:
	SubOutputNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
	~SubOutputNode();

protected:
	void onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue) override;
};

#endif