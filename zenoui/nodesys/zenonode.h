#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include "../render/renderparam.h"
#include "zenosvgitem.h"

class ZenoNode : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;
    void init(const QModelIndex& index);
    void initParams(int& y);
    void initSockets(int& y);
    QPersistentModelIndex index() { return m_index; }
   
    QPointF getPortPos(bool bInput, const QString& portName);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    QJsonObject inputParams() const;

signals:
    void nodePositionChange(const QString&);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

private:
    QPersistentModelIndex m_index;
    NodeUtilParam m_renderParams;
    std::map<QString, ZenoImageItem*> m_inSocks;
    std::map<QString, ZenoImageItem*> m_outSocks;

    QGraphicsTextItem* m_nameItem;
    ZenoImageItem *m_headerBg;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_prep;
    ZenoImageItem *m_collaspe;
    ZenoImageItem *m_bodyBg;
};

#endif