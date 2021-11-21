#ifndef __NODEGRID_H__
#define __NODEGRID_H__

class NodeGridItem;

class NodeGridLineItem : public QGraphicsLineItem
{
public:
    NodeGridLineItem(NodeGridItem* grid, qreal x1, qreal y1, qreal x2, qreal y2, QGraphicsItem* parent = nullptr);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget);

private:
    NodeGridItem* m_grid;
};

class NodeGridItem : public QGraphicsRectItem
{
public:
    NodeGridItem(QGraphicsItem* parent = nullptr);
    void setFactor(qreal factor) { m_factor = factor; }
    qreal factor() const { return m_factor; }

private:
    qreal m_factor;

    int m_nLargeCellRows;
    int m_nLargeCellColumns;
    int m_nCellsInLargeCells;
    int m_nPixelsInCell;
};

#endif