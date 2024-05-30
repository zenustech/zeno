#ifndef __GROUP_NODE_H__
#define __GROUP_NODE_H__

#include "zenonodebase.h"

class GroupTextItem : public QGraphicsWidget {
    Q_OBJECT
public:
    GroupTextItem(QGraphicsItem *parent = nullptr);
    ~GroupTextItem();
    void setText(const QString &text);
    QString text() const;

signals:
    void mousePressSignal(QGraphicsSceneMouseEvent *event);
    void mouseMoveSignal(QGraphicsSceneMouseEvent *event);
    void mouseReleaseSignal(QGraphicsSceneMouseEvent *event);
    void textChangedSignal(const QString &text);

protected: 
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

private:
    QString m_text;
    ZEditableTextItem* m_pLineEdit;
};


class GroupNode : public ZenoNodeBase {
    Q_OBJECT
        typedef ZenoNodeBase _base;
public:
    GroupNode(const NodeUtilParam &params, QGraphicsItem *parent = nullptr);
    ~GroupNode();
    bool nodePosChanged(ZenoNodeBase*);
    void onZoomed() override;
    QRectF boundingRect() const override;
    void appendChildItem(ZenoNodeBase*item);
    void updateChildItemsPos();
    QVector<ZenoNodeBase*> getChildItems();
    void removeChildItem(ZenoNodeBase*pNode);
    void updateChildRelativePos(const ZenoNodeBase*item);
    void updateBlackboard();
    void setSelected(bool selected) override;
    void initLayout() override;

protected:

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
private slots:
    void onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles);
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

    bool m_bDragging;
    bool m_bSelected;
    QVector<ZenoNodeBase*> m_childItems;
    GroupTextItem *m_pTextItem;
    QMap<QString, QPointF> m_itemRelativePosMap;
    QByteArray m_svgByte;
};


#endif