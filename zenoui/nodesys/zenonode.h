#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include "../render/renderparam.h"
#include "zenosvgitem.h"
#include "zenobackgrounditem.h"
#include "nodesys_common.h"
#include "zenosocketitem.h"
#include "../model/modeldata.h"


class ZenoNode : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;

public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    ~ZenoNode();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;

    enum { Type = ZTYPE_NODE };
    int type() const override;

    void init(const QModelIndex& index);
    void initParams(int& y, int& width);
    void initSockets(int& y, int& width);

    QPersistentModelIndex index() { return m_index; }
    QPointF getPortPos(bool bInput, const QString& portName);

    QString nodeId() const;
    QString nodeName() const;
    QPointF nodePos() const;
    INPUT_SOCKETS inputParams() const;
    OUTPUT_SOCKETS outputParams() const;

signals:
    void nodePositionChange(const QString&);
    void socketClicked(const QString& id, bool bInput, const QString& name);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    bool sceneEventFilter(QGraphicsItem* watched, QEvent* event) override;
    bool sceneEvent(QEvent* event) override;

private:
    QPersistentModelIndex m_index;
    NodeUtilParam m_renderParams;
    std::map<QString, ZenoSocketItem *> m_inSocks;
    std::map<QString, ZenoSocketItem *> m_outSocks;

    QGraphicsTextItem* m_nameItem;
    ZenoBackgroundItem *m_headerBg;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_prep;
    ZenoImageItem *m_collaspe;
    ZenoBackgroundItem *m_bodyBg;
};

#endif