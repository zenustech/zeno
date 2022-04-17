#ifndef __NODES_SCENE_H__
#define __NODES_SCENE_H__

#include <rapidxml/rapidxml_print.hpp>
#include <render/renderparam.h>

class TimelineItem;
class DragPointItem;
class ComponentItem;
class NodeGridItem;
class NodeTemplate;
class NodesView;

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
    NodeScene(NodesView* pView, QObject* parent = nullptr);
    ~NodeScene();
	void initGrid();
	void initTimelines(QRectF rcView);
	void initSkin(const QString& fn);
	void initNode();
    QSizeF getSceneSize();
    QStandardItemModel* model() const;
    QItemSelectionModel* selectionModel() const;
    NodeParam exportNodeParam();

public slots:
    void updateDragPoints(QGraphicsItem* pDragged, DRAG_ITEM dragWay);
    void onSelectionChanged();
    void _adjustDragRectPos(QGraphicsItem* pSelection);
    void updateTimeline(qreal factor);
    void onViewTransformChanged(qreal factor);
    void resetPreset(int W, int H);

signals:
    void markDirty();

private:
	QVector<DragPointItem*> m_dragPoints;
	QGraphicsRectItem* m_selectedRect;
    QGraphicsItem* m_selectedItem;
    NodeGridItem* m_grid;
    TimelineItem* m_pHTimeline, *m_pVTimeline;
    NodeTemplate* m_pNode;

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