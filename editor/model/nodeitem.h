#ifndef __NODEITEM_H__
#define __NODEITEM_H__

#include "../framework.h"

struct SocketItem
{
	QString name;
};

struct NodeItem
{
	QString name;
	QString id;
	QRectF sceneRect;
	std::vector<SocketItem> inSockets;
};

struct LinkItem
{
	QString id;
	QString srcNodeId;
	QString dstNodeId;
};

#endif