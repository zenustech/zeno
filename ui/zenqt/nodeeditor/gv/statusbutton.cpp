#include "statusbutton.h"
#include "util/uihelper.h"


StatusButton::StatusButton(RoundRectInfo info, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_bOn(false)
    , m_bHovered(false)
    , m_info(info)
{
    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable);
    initPath();
}

void StatusButton::setColor(bool bOn, QColor clrOn, QColor clrOff)
{
    m_bOn = bOn;
    m_clrOff = clrOff;
    m_clrOn = clrOn;
}

QRectF StatusButton::boundingRect() const
{
    return QRectF(0, 0, m_info.W, m_info.H);
}

void StatusButton::initPath()
{
    QRectF rc(0, 0, m_info.W, m_info.H);
    m_path = UiHelper::getRoundPath(rc, m_info.ltradius, m_info.rtradius, m_info.lbradius, m_info.rbradius, true);
}

QPainterPath StatusButton::shape() const
{
    return m_path;
}

void StatusButton::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    painter->setRenderHint(QPainter::Antialiasing, true);
    if (m_bHovered || m_bOn)
        painter->fillPath(m_path, m_clrOn);
    else
        painter->fillPath(m_path, m_clrOff);
}

void StatusButton::setHovered(bool bHovered)
{
    m_bHovered = bHovered;
    update();
}

void StatusButton::toggle(bool bSelected)
{
    if (bSelected == m_bOn)
        return;

    m_bOn = bSelected;
    if (!m_bOn) {
        m_bHovered = false;
    }
    emit toggled(m_bOn);
    update();
}

void StatusButton::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverEnterEvent(event);
    m_bHovered = true;
}

void StatusButton::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverLeaveEvent(event);
}

void StatusButton::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverLeaveEvent(event);
    m_bHovered = false;
}

void StatusButton::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
    event->setAccepted(true);
}

void StatusButton::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    toggle(!m_bOn);
}