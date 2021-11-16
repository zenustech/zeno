#ifndef __NODES_SCENE_H__
#define __NODES_SCENE_H__

#include <rapidxml/rapidxml_print.hpp>
#include "renderparam.h"

class NodeScene : public QGraphicsScene
{
	Q_OBJECT
public:
	NodeScene(QObject* parent = nullptr);
	void initGrid();
	void initSkin(const QString& fn);
	void initNode();

private:
	NodeParam m_nodeparam;
	int m_nLargeCellRows;
	int m_nLargeCellColumns;
	int m_nCellsInLargeCells;
	int m_nPixelsInCell;
};

#endif