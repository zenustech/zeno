#ifndef __NODEITEM_H__
#define __NODEITEM_H__

#include <QString>
#include <QRectF>
#include <vector>
#include <QtWidgets>

struct SocketItem
{
	QString name;
};

struct ParamsItem {
};

struct NodeItem
{
    typedef std::unordered_map<QString, NodeItem *> MAPPER;

	QString name;
	QString id;
	QRectF sceneRect;
    QJsonObject params;
    QJsonObject inputs;
    QJsonObject outputs;
	NodeItem* parent;
	std::unordered_map<QString, NodeItem*> m_childrens;
};

struct LinkItem
{
	QString id;
	QString srcNodeId;
	QString dstNodeId;
};

#endif
