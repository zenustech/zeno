#ifndef __ZENO_HEATMAPITEM_H__
#define __ZENO_HEATMAPITEM_H__

#include <QtWidgets>
#include "zenoparamwidget.h"
#include "../model/modeldata.h"

class ZenoColorChannelItem;
class ZenoColorRampItem;
class ZenoHeatMapItem;

class ZenoItemNoDragThrough : public QGraphicsItem
{
public:
    ZenoItemNoDragThrough(QGraphicsItem *parent = nullptr);
    enum { Type = ZTYPE_NODRAGITEM };
    int type() const { return Type; }

protected:
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event);
};

class ZenoRampDraggerItem : public ZenoItemNoDragThrough
{
    typedef ZenoItemNoDragThrough _base;
public:
    ZenoRampDraggerItem(QGraphicsItem *parent = nullptr);
    qreal width() const;
    qreal height() const;
    QRectF boundingRect() const override;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    qreal getValue();
    void setValue(qreal x);
    void setX(qreal x);
    void incX(qreal dx);
    void setSelected(bool selected);
    void remove();
    bool IsSelected() const { return m_selected; }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;

private:
    QRectF parentRect() const;

    QGraphicsItem *m_parent;
    bool m_selected;
};

class ZenoColorChannelItem : public QGraphicsLayoutItem
                           , public ZenoItemNoDragThrough
{
public:
    ZenoColorChannelItem(ZenoHeatMapItem* parent = nullptr);
    void setGeometry(const QRectF& rect) override;
    void setColor(qreal r, qreal g, qreal b);
    QRectF boundingRect() const override;

    enum { Type = ZTYPE_COLOR_CHANNEL };
    int type() const { return Type; }

    qreal getValue();
    void setValue(qreal x);
    void updateRamps();
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QRectF rect() const;

protected:
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF &constraint = QSizeF()) const override;

private:
    QRectF m_rect;
    QColor m_color;
    ZenoRampDraggerItem* m_dragger;
    ZenoHeatMapItem* m_parent;
};

class ZenoColorRampItem : public QGraphicsLayoutItem
                        , public ZenoItemNoDragThrough
{
public:
    ZenoColorRampItem(ZenoHeatMapItem* parent = nullptr);
    void updateRampSelection(ZenoRampDraggerItem* this_dragger);
    int currSelectedIndex();
    void updateRampColor(qreal r, qreal g, qreal b);
    void removeRamp(ZenoRampDraggerItem *dragger);
    void addRampAt(qreal fac);
    void initDraggers();
    void updateRamps();

    COLOR_RAMPS& ramps();
    void setGeometry(const QRectF& rect) override;
    QRectF boundingRect() const override;
    QRectF rect() const;

    enum { Type = ZTYPE_COLOR_RAMP };
    int type() const { return Type; }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = nullptr) override;
    QSizeF sizeHint(Qt::SizeHint which, const QSizeF& constraint = QSizeF()) const override;

private:
    QRectF m_rect;
    ZenoHeatMapItem* m_parent;
    QList<ZenoRampDraggerItem*> m_draggers;
};


class ZenoHeatMapItem : public ZenoParamWidget
{
    Q_OBJECT
public:
    ZenoHeatMapItem(const COLOR_RAMPS& ramps, QGraphicsItem *parent = nullptr);
    void updateRampColor();
    void updateRampSelection();
    void dump();
    void load(const QString& id, const COLOR_RAMPS& colorRamps);
    COLOR_RAMPS& ramps();
    void setColorRamps(const COLOR_RAMPS &ramps);

    enum { Type = ZTYPE_HEATMAP };
    int type() const { return Type; }

private:
    void initWidgets();

    COLOR_RAMPS m_colorRamps;
    ZenoColorRampItem* m_colorramp;
    ZenoColorChannelItem* m_colorR;
    ZenoColorChannelItem* m_colorG;
    ZenoColorChannelItem* m_colorB;
};


#endif