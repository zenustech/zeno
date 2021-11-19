#ifndef __COMPONENT_H__
#define __COMPONENT_H__

class NodeScene;

class ComponentItem : QGraphicsRectItem
{
    typedef QGraphicsRectItem _base;
public:
    ComponentItem(NodeScene* pScene, qreal x, qreal y, qreal w, qreal h, QGraphicsItem* parent = nullptr);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget /* = nullptr */);

private:
    NodeScene* m_pScene;
};

#endif