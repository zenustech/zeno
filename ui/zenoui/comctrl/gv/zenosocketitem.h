#ifndef __ZENO_SOCKET_ITEM_H__
#define __ZENO_SOCKET_ITEM_H__

#include <zenoui/model/modeldata.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include "../../nodesys/nodesys_common.h"

class ZenoSocketItem : public ZenoImageItem
{
    Q_OBJECT
    typedef ZenoImageItem _base;

public:
    enum SOCK_STATUS
    {
        STATUS_UNKNOWN,
        STATUS_NOCONN,
        STATUS_TRY_CONN,
        STATUS_TRY_DISCONN,
        STATUS_CONNECTED,
    };

    ZenoSocketItem(const ImageElement &elem, const QSizeF &sz, QGraphicsItem *parent = 0);
    enum { Type = ZTYPE_SOCKET };
    int type() const override;
    void setOffsetToName(const QPointF& offsetToName);
    QRectF boundingRect() const override;
    void setIsInput(bool left);
    bool isInput() const;
    void setSockStatus(SOCK_STATUS status);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;

public slots:
    void socketNamePosition(const QPointF& nameScenePos);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;

private:
    QPointF m_offsetToName;
    SOCK_STATUS m_status;
    ZenoSvgItem* m_svgHover;
    QString m_noHoverSvg;
    QString m_hoverSvg;
    bool m_bLeftSock;
};

#endif