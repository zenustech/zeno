#ifndef __STATUS_GROUP_H__
#define __STATUS_GROUP_H__

#include <QtWidgets>
#include "uicommon.h"
#include "nodeeditor/gv/nodesys_common.h"
#include "zlayoutbackground.h"

#include <QGraphicsItemGroup>
#include <QGraphicsItem>
#include <QPainter>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneMouseEvent>


class CommonStatusBtn : public QGraphicsObject
{
    Q_OBJECT
    typedef QGraphicsObject _base;
public:
    CommonStatusBtn(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    void updateHasRadiusOnButtom(bool bOn);

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

    RoundRectInfo m_rectInfo;
    bool m_bOn;
    bool m_bHovered;
    bool m_bHasRadiusOnButtom;
};


class ByPassButton : public CommonStatusBtn
{
public:
    ByPassButton(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;
};

class ViewButton : public CommonStatusBtn
{
public:
    ViewButton(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;
};

class NoCacheButton : public CommonStatusBtn
{
public:
    NoCacheButton(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;
};

class ClearSbnCacheButton : public CommonStatusBtn
{
public:
    ClearSbnCacheButton(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*) override;
};


class LeftStatusBtnGroup : public ZGraphicsLayoutItem<QGraphicsWidget>
{
    Q_OBJECT
    typedef ZGraphicsLayoutItem<QGraphicsWidget> _base;
public:
    LeftStatusBtnGroup(zeno::NodeType type, RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override {}
    void setNoCache(bool nocache);
    void setClearSubnet(bool clearSbn);
    void updateRightButtomRadius(bool bHasRadius);

signals:
    void toggleChanged(STATUS_BTN btn, bool hovered);

private:
    NoCacheButton* m_nocache;
    ClearSbnCacheButton* m_clearsubnet;
    RoundRectInfo m_rectInfo;
};

class RightStatusBtnGroup : public ZGraphicsLayoutItem<QGraphicsWidget>
{
    Q_OBJECT
    typedef ZGraphicsLayoutItem<QGraphicsWidget> _base;

public:
    RightStatusBtnGroup(RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override {}
    void setView(bool isView);
    void setByPass(bool bypass);
    void updateRightButtomRadius(bool bHasRadius);

signals:
    void toggleChanged(STATUS_BTN btn, bool hovered);

private:
    ByPassButton* m_bypass;
    ViewButton* m_view;
    RoundRectInfo m_rectInfo;
};


class StatusButton;
class ZenoImageItem;

class StatusGroup : public ZLayoutBackground
{
    Q_OBJECT
    typedef ZLayoutBackground _base;

public:
    StatusGroup(bool bHasOptimStatus, RoundRectInfo info, QGraphicsItem* parent = nullptr);
    QRectF boundingRect() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override;
    void setChecked(STATUS_BTN btn, bool bChecked);
    void setOptions(int options);
    void setView(bool isView);
    void updateRightButtomRadius(bool bHasRadius);
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
    ZenoImageItem* m_optim;
    StatusButton* m_minMute;
    StatusButton* m_minView;
    StatusButton* m_minOptim;
};

#endif