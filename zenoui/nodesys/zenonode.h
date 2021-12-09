#ifndef __ZENO_NODE_H__
#define __ZENO_NODE_H__

#include <QtWidgets>
#include "../render/renderparam.h"
#include "zenosvgitem.h"

class ZenoNode : public QGraphicsItem
{
public:
    ZenoNode(const NodeUtilParam& params, QGraphicsItem *parent = nullptr);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF boundingRect() const override;
    void init(const QModelIndex& index);
    void initParams(int& y);
    void initSockets(int& y);
    QPersistentModelIndex index() { return m_index; }

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;

private:
    QPersistentModelIndex m_index;
    NodeUtilParam m_renderParams;
    std::vector<ZenoImageItem*> m_InSockets;
    std::vector<ZenoImageItem*> m_OutSockets;

    QGraphicsTextItem* m_nameItem;
    ZenoImageItem *m_headerBg;
    ZenoImageItem *m_mute;
    ZenoImageItem *m_view;
    ZenoImageItem *m_prep;
    ZenoImageItem *m_collaspe;
    ZenoImageItem *m_bodyBg;
};

#endif