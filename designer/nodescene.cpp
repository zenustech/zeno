#include "framework.h"
#include "nodescene.h"
#include "zenonode.h"

NodeScene::NodeScene(QObject* parent)
	: QGraphicsScene(parent)
{
	initNode();
}

void NodeScene::initNode()
{
	ZenoNode* pNode = new ZenoNode;
	addItem(pNode);
	pNode->setPos(QPointF(0, 0));
	pNode->show();
}