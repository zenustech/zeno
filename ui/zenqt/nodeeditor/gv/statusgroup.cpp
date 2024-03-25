#include "statusgroup.h"
#include "statusbutton.h"
#include "zenosvgitem.h"
#include "style/zenostyle.h"


StatusGroup::StatusGroup(qreal W, qreal H, qreal rtradius, qreal rbradius, QGraphicsItem* parent)
    : ZLayoutBackground(parent)
{
    setColors(false, QColor(0, 0, 0, 0));

    RoundRectInfo rectInfo, roundInfo;
    rectInfo.W = W;
    rectInfo.H = H;
    roundInfo.W = W;
    roundInfo.H = H;
    roundInfo.rtradius = rtradius;
    roundInfo.rbradius = rbradius;

    m_minMute = new StatusButton(rectInfo);
    m_minMute->setColor(false, QColor("#E302F8"), QColor("#2F3135"));

    m_minView = new StatusButton(roundInfo);
    m_minView->setColor(false, QColor("#26C5C5"), QColor("#2F3135"));

    ZGraphicsLayout* pLayout = new ZGraphicsLayout(true);
    pLayout->setSpacing(1);
    pLayout->addItem(m_minMute);
    pLayout->addItem(m_minView);
    this->setLayout(pLayout);

    m_mute = new ZenoImageItem(
        ":/icons/MUTE_dark.svg",
        ":/icons/MUTE_light.svg",
        ":/icons/MUTE_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);

    m_view = new ZenoImageItem(
        ":/icons/VIEW_dark.svg",
        ":/icons/VIEW_light.svg",
        ":/icons/VIEW_light.svg",
        ZenoStyle::dpiScaledSize(QSize(50, 42)),
        this);
    m_mute->setCheckable(true);
    m_view->setCheckable(true);
    m_mute->hide();
    m_view->hide();

    QSizeF sz2 = m_mute->size();
    qreal sMarginTwoBar = ZenoStyle::dpiScaled(4);
    //todo: kill these magin number.
    QPointF base = QPointF(0, -sz2.height() - sMarginTwoBar);
    m_mute->setPos(base);
    base += QPointF(ZenoStyle::dpiScaled(38), 0);
    m_view->setPos(base);

    connect(m_minView, SIGNAL(hoverChanged(bool)), m_view, SLOT(setHovered(bool)));
    connect(m_minMute, SIGNAL(hoverChanged(bool)), m_mute, SLOT(setHovered(bool)));

    connect(m_view, SIGNAL(hoverChanged(bool)), m_minView, SLOT(setHovered(bool)));
    connect(m_mute, SIGNAL(hoverChanged(bool)), m_minMute, SLOT(setHovered(bool)));

    connect(m_minView, SIGNAL(toggled(bool)), m_view, SLOT(toggle(bool)));
    connect(m_minMute, SIGNAL(toggled(bool)), m_mute, SLOT(toggle(bool)));

    connect(m_view, SIGNAL(toggled(bool)), m_minView, SLOT(toggle(bool)));
    connect(m_mute, SIGNAL(toggled(bool)), m_minMute, SLOT(toggle(bool)));

    connect(m_minMute, &StatusButton::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_MUTE, hovered);
        });
    connect(m_minView, &StatusButton::toggled, [=](bool hovered) {
        emit toggleChanged(STATUS_VIEW, hovered);
        });
}

QRectF StatusGroup::boundingRect() const
{
    return _base::boundingRect();
}

void StatusGroup::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    _base::paint(painter, option, widget);
}

void StatusGroup::setChecked(STATUS_BTN btn, bool bChecked)
{

}

void StatusGroup::setOptions(int options)
{

}

void StatusGroup::setView(bool isView)
{

}

void StatusGroup::onZoomed()
{

}

void StatusGroup::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_mute->show();
    m_view->show();
    _base::hoverEnterEvent(event);
}

void StatusGroup::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void StatusGroup::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    m_mute->hide();
    m_view->hide();
    _base::hoverLeaveEvent(event);
}
