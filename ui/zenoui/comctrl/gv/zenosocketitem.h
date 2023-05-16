#ifndef __ZENO_SOCKET_ITEM_H__
#define __ZENO_SOCKET_ITEM_H__

#include <zenomodel/include/modeldata.h>
#include <zenoui/nodesys/zenosvgitem.h>
#include "../../nodesys/nodesys_common.h"

class ZenoSocketItem : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;
public:
    enum SOCK_STATUS
    {
        STATUS_UNKNOWN,
        STATUS_NOCONN,
        STATUS_TRY_CONN,
        STATUS_TRY_DISCONN,
        STATUS_CONNECTED,
    };

    ZenoSocketItem(
        const QPersistentModelIndex& viewSockIdx,
        const QSizeF& sz,
        QGraphicsItem *parent = 0);
    enum { Type = ZTYPE_SOCKET };
    int type() const override;
    QRectF boundingRect() const override;
    QPointF center() const;
    QModelIndex paramIndex() const;
    bool isInputSocket() const;
    QString nodeIdent() const;
    void setSockStatus(SOCK_STATUS status);
    void setHovered(bool bHovered);
    SOCK_STATUS sockStatus() const;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;

signals:
    void clicked(bool bInput);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    SOCK_STATUS m_status;
    const QPersistentModelIndex m_viewSockIdx;
    QSizeF m_size;
    int m_innerSockMargin;
    int m_socketXOffset;
    bool m_bInput;
    bool m_bInnerSock;
    bool m_bHovered;
};

#endif