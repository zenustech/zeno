#ifndef __NODES_SCENE_H__
#define __NODES_SCENE_H__

#include <rapidxml/rapidxml_print.hpp>
#include "renderparam.h"

class TimelineItem;
class DragPointItem;
class ComponentItem;

class NodeScene : public QGraphicsScene
{
	Q_OBJECT

public:
    enum DRAG_ITEM
    {
        DRAG_LEFTTOP,
        DRAG_LEFTMID,
        DRAG_LEFTBOTTOM,
        DRAG_MIDTOP,
        DRAG_MIDBOTTOM,
        DRAG_RIGHTTOP,
        DRAG_RIGHTMID,
        DRAG_RIGHTBOTTOM,
        TRANSLATE,
        SELECT,
        UNSELECT,
        NO_DRAG,
    };

public:
	NodeScene(QObject* parent = nullptr);
	void initGrid();
	void initTimelines();
	void initSkin(const QString& fn);
	void initNode();
	void initSelectionDragBorder();

public slots:
	void timelineChanged();
    void updateDragPoints(QGraphicsItem* pDragged, DRAG_ITEM dragWay);
    void onSelectionChanged();
    void _adjustDragRectPos(QGraphicsItem* pSelection);

private:
	TimelineItem* m_pHTimeline;

	QVector<DragPointItem*> m_dragPoints;
	QGraphicsRectItem* m_selectedRect;
    QGraphicsItem* m_selectedItem;

	NodeParam m_nodeparam;
	int m_nLargeCellRows;
	int m_nLargeCellColumns;
	int m_nCellsInLargeCells;
	int m_nPixelsInCell;

    const qreal dragW = 8.;
    const qreal dragH = 8.;
    const qreal borderW = 1.;
};

#endif