#ifndef __BLACKBOARD_NODE2_H__
#define __BLACKBOARD_NODE2_H__

#include "zenonode.h"

class GroupTextItem : public QGraphicsWidget {
    Q_OBJECT
  public:
    GroupTextItem(QGraphicsItem *parent = nullptr);
    ~GroupTextItem();
    void setText(const QString &text);

  signals:
    void updatePosSignal();
    void posChangedSignal(const QPointF &pos);
    void mousePressSignal();

  protected: 
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

  private:
    bool m_bMoving;
    QString m_text;
    QPointF m_beginPos;
};


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
    void updateChildRelativePos(const ZenoNode *item);
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
    QRectF getSelectArea();
    void setSvgData(QString color);
    enum {
    	nodir,
    	top = 0x01,
    	bottom = 0x02,
    	left = 0x04,
    	right = 0x08,
    	topLeft = 0x01 | 0x04,
    	topRight = 0x01 | 0x08,
    	bottomLeft = 0x02 | 0x04,
    	bottomRight = 0x02 | 0x08
    } resizeDir;
  private:
    bool m_bDragging;
    bool m_bSelecting;
    QVector<ZenoNode *> m_childItems;
    QPointF m_beginPos;
    QPointF m_endPos;
    GroupTextItem *m_pTextItem;
    QMap<QString, QPointF> m_itemRelativePosMap;
    QByteArray m_svgByte;
};


#endif