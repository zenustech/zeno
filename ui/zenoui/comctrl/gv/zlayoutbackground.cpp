#include "zlayoutbackground.h"
#include "../../style/zenostyle.h"
#include <QtWidgets>
#include <zenomodel/include/uihelper.h>



///////////////////////////////////////////////////////////////////////////
ZLayoutBackground::ZLayoutBackground(QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : _base(parent)
    , lt_radius(0)
    , rt_radius(0)
    , lb_radius(0)
    , rb_radius(0)
    , m_borderWidth(0)
    , m_bFixRadius(true)
    , m_bSelected(false)
{
    setAcceptHoverEvents(true);
}

ZLayoutBackground::~ZLayoutBackground()
{
}

QRectF ZLayoutBackground::boundingRect() const
{
    QRectF rc = _base::boundingRect();
    qreal halfpw = (qreal)m_borderWidth / 2;
    //if (halfpw > 0.0)
 //       rc.adjust(-halfpw, -halfpw, halfpw, halfpw);
    return rc;
}

void ZLayoutBackground::setBorder(qreal width, const QColor& clrBorder)
{
    m_borderWidth = width;
    m_clrBorder = clrBorder;
}

void ZLayoutBackground::setColors(bool bAcceptHovers, const QColor& clrNormal, const QColor& clrHovered, const QColor& clrSelected)
{
    setAcceptHoverEvents(bAcceptHovers);
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    m_color = m_clrNormal;
    update();
}

void ZLayoutBackground::setRadius(int lt, int rt, int lb, int rb)
{
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

QSizeF ZLayoutBackground::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    return QGraphicsWidget::sizeHint(which, constraint);
}

void ZLayoutBackground::setGeometry(const QRectF& rect)
{
    QGraphicsWidget::setGeometry(rect);
}

void ZLayoutBackground::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (m_borderWidth > 0)
    {
        painter->setRenderHint(QPainter::Antialiasing, true);
        QPainterPath path = shape();
        QPen pen(m_clrBorder, m_borderWidth);
        pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);
        painter->setBrush(m_color);
        painter->drawPath(path);
    }
    else
    {
        QPainterPath path = shape();
        painter->fillPath(path, m_color);
    }
}

void ZLayoutBackground::toggle(bool bSelected)
{
    m_bSelected = bSelected;
    if (m_clrSelected.isValid())
    {
        m_color = m_bSelected ? m_clrSelected : m_clrNormal;
        update();
    }
}

QPainterPath ZLayoutBackground::shape() const
{
    QRectF r = boundingRect();
    r.adjust(m_borderWidth / 2, m_borderWidth / 2, -m_borderWidth / 2, -m_borderWidth / 2);
    return UiHelper::getRoundPath(r, lt_radius, rt_radius, lb_radius, rb_radius, m_bFixRadius);
}

void ZLayoutBackground::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverEnterEvent(event);
    if (m_clrHovered.isValid())
    {
        m_color = m_clrHovered;
        update();
    }
    emit hoverEntered();
}

void ZLayoutBackground::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZLayoutBackground::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverLeaveEvent(event);
    if (m_clrSelected.isValid())
    {
        m_color = isSelected() ? m_clrSelected : m_clrNormal;
        update();
    }
    emit hoverLeaved();
}

void ZLayoutBackground::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
}

void ZLayoutBackground::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
}
