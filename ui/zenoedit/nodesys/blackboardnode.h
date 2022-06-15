#ifndef __BLACKBOARD_NODE_H__
#define __BLACKBOARD_NODE_H__

#include "zenonode.h"

class BlackboardNode : public ZenoNode
{
    Q_OBJECT
public:
    BlackboardNode(const NodeUtilParam& params, QGraphicsItem* parent = nullptr);
    ~BlackboardNode();
    QRectF boundingRect() const override;

protected:
    ZenoBackgroundWidget* initBodyWidget(NODE_TYPE type) override;
    ZenoBackgroundWidget* initHeaderWangStyle(NODE_TYPE type) override;

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;

private slots:
    void updateBlackboard();

private:
    bool isDragArea(QPointF pos);

    ZenoParamBlackboard* m_pTextEdit;
    ZenoTextLayoutItem* m_pTitle;
    bool m_bDragging;
};


#endif