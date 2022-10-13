#ifndef __ZENO_SOCKET_ITEM_H__
#define __ZENO_SOCKET_ITEM_H__

#include <zenomodel/include/modeldata.h>
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

    ZenoSocketItem(
        const QString& sockName,
        bool bInput,
        QPersistentModelIndex nodeIdx,
        const ImageElement &elem,
        const QSizeF &sz,
        QGraphicsItem *parent = 0);
    enum { Type = ZTYPE_SOCKET };
    int type() const override;
    void setOffsetToName(const QPointF& offsetToName);
    void setup(const QModelIndex& idx);
    QRectF boundingRect() const override;
    QPointF center() const;
    bool getSocketInfo(bool& bInput, QString& nodeid, QString& sockName);
    void updateSockName(const QString& sockName);
    void setSockStatus(SOCK_STATUS status);
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0) override;

signals:
    void clicked(bool bInput);

public slots:
    void socketNamePosition(const QPointF& nameScenePos);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    QPointF m_offsetToName;
    SOCK_STATUS m_status;
    ZenoSvgItem* m_svgHover;
    QString m_noHoverSvg;
    QString m_hoverSvg;

    QPersistentModelIndex m_index;
    QString m_name;         //should update when meet dynamic socket.

    const int sHorLargeMargin;
    const int sTopMargin;
    const int sHorSmallMargin;
    const int sBottomMargin;

    const bool m_bInput;
};

#endif