#ifndef __NODES_SCENE_H__
#define __NODES_SCENE_H__

class NodeScene : public QGraphicsScene
{
	Q_OBJECT
public:
	NodeScene(QObject* parent = nullptr);
	void initGrid();

private:
	void initNode();

	int m_nLargeCellRows;
	int m_nLargeCellColumns;
	int m_nCellsInLargeCells;
	int m_nPixelsInCell;

};

#endif