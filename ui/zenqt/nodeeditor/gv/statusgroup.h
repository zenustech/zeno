#ifndef __STATUS_GROUP_H__
#define __STATUS_GROUP_H__

#include <QtWidgets>
#include "uicommon.h"
#include "nodeeditor/gv/nodesys_common.h"
#include "zlayoutbackground.h"

class StatusButton;
class ZenoImageItem;

class StatusGroup : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;

public:
    StatusGroup(qreal W, qreal H, qreal rtradius, qreal rbradius, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void setChecked(STATUS_BTN btn, bool bChecked);
    void setOptions(int options);
    void setView(bool isView);
    void onZoomed();

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

signals:
    void toggleChanged(STATUS_BTN btn, bool hovered);

protected:
    ZenoImageItem* m_mute;
    ZenoImageItem* m_view;
    StatusButton* m_minMute;
    StatusButton* m_minView;
};

#endif