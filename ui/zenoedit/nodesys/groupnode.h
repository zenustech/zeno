#ifndef __GROUP_NODE_H__
#define __GROUP_NODE_H__

#include "zenonode.h"

class GroupTextItem : public QGraphicsWidget {
    Q_OBJECT
  public:
    GroupTextItem(QGraphicsItem *parent = nullptr);
    ~GroupTextItem();
    void setText(const QString &text);

  signals:
    void mousePressSignal(QGraphicsSceneMouseEvent *event);
    void mouseMoveSignal(QGraphicsSceneMouseEvent *event);
    void mouseReleaseSignal(QGraphicsSceneMouseEvent *event);

  protected: 
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

  private:
    QString m_text;
};


class GroupNode : public ZenoNode {
    Q_OBJECT
  public:
    GroupNode(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~GroupNode();
    bool nodePosChanged(ZenoNode *);
    void onZoomed() override;
    QRectF boundingRect() const override;
    void onUpdateParamsNotDesc() override;
    void appendChildItem(ZenoNode *item);
    void updateChildItemsPos();
    QVector<ZenoNode *> getChildItems();
    void removeChildItem(ZenoNode *pNode);
    void updateChildRelativePos(const ZenoNode *item);
    void updateBlackboard();

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
    void updateClidItem(bool isAdd, const QString nodeId);
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
    bool m_bSelected;
    QVector<ZenoNode *> m_childItems;
    GroupTextItem *m_pTextItem;
    QMap<QString, QPointF> m_itemRelativePosMap;
    QByteArray m_svgByte;
};


#endif