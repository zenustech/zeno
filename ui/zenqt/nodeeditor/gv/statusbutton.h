#ifndef __STATUS_BUTTON_H__
#define __STATUS_BUTTON_H__

#include <QtWidgets>
#include "uicommon.h"
#include "nodeeditor/gv/nodesys_common.h"

class StatusButton : public QGraphicsObject
{
    Q_OBJECT
        typedef QGraphicsObject _base;
public:
    StatusButton(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    void setColor(bool bOn, QColor clrOn, QColor clrOff);
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void updateRightButtomRadius(bool bHasRadius);

    static const int dirtyLayoutHeight = 2;

signals:
    void hoverChanged(bool);
    void toggled(bool);

public slots:
    void setHovered(bool bHovered);
    void toggle(bool bSelected);

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    void initPath();

    QPainterPath m_path;
    RoundRectInfo m_info;
    bool m_bOn;
    bool m_bHovered;
    QColor m_clrOn;
    QColor m_clrOff;
};


#endif