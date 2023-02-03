#ifndef __BLACKBOARD_NODE2_H__
#define __BLACKBOARD_NODE2_H__

#include "zenonode.h"

class BlackboardNode2 : public ZenoNode {
    Q_OBJECT
  public:
    BlackboardNode2(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~BlackboardNode2();
    bool nodePosChanged(ZenoNode *);
    void onZoomed() override;
    QRectF boundingRect() const override;
    void onUpdateParamsNotDesc() override;
    void appendChildItem(ZenoNode *item);
    void updateChildItemsPos();
    QVector<ZenoNode *> getChildItems();
    void removeChildItem(ZenoNode *pNode);
  protected:
    ZLayoutBackground *initBodyWidget(ZenoSubGraphScene *pScene) override;
    ZLayoutBackground *initHeaderWidget(IGraphsModel*) override;

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

  private:
    bool isDragArea(QPointF pos);
    void updateBlackboard();
    void updateClidItem(bool isAdd, const QString nodeId);
  private:
    bool m_bDragging;
    QVector<ZenoNode *> m_childItems;
};


#endif