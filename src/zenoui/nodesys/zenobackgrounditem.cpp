#include "zenobackgrounditem.h"
#include "../util/uihelper.h"


ZenoBackgroundItem::ZenoBackgroundItem(const BackgroundComponent &comp, QGraphicsItem *parent)
    : _base(parent)
    , m_rect(QRectF(0, 0, comp.rc.width(), comp.rc.height()))
    , lt_radius(comp.lt_radius)
    , rt_radius(comp.rt_radius)
    , lb_radius(comp.lb_radius)
    , rb_radius(comp.rb_radius)
    , m_bFixRadius(true)
    , m_img(nullptr)
    , m_bSelected(false)
{
    if (!comp.imageElem.image.isEmpty())
    {
        m_img = new ZenoImageItem(comp.imageElem.image, comp.imageElem.imageHovered, comp.imageElem.imageOn, comp.rc.size(), this);
        m_img->setZValue(100);
        m_img->show();
    }
    setColors(comp.clr_normal, comp.clr_hovered, comp.clr_selected);
}

QRectF ZenoBackgroundItem::boundingRect() const {
    return m_rect;
}

void ZenoBackgroundItem::resize(QSizeF sz) {
    QPointF topLeft = m_rect.topLeft();
    m_rect = QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height());
    if (m_img)
        m_img->resize(sz);
}

void ZenoBackgroundItem::setColors(const QColor &clrNormal, const QColor &clrHovered, const QColor &clrSelected) {
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    update();
}

void ZenoBackgroundItem::setRadius(int lt, int rt, int lb, int rb) {
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

QPainterPath ZenoBackgroundItem::shape() const
{
    QPainterPath path;
    QRectF r = m_rect.normalized();
    return UiHelper::getRoundPath(r, lt_radius, rt_radius, lb_radius, rb_radius, m_bFixRadius);
}

void ZenoBackgroundItem::toggle(bool bSelected)
{
    if (m_img) {
        m_img->toggle(bSelected);
    } else {
        m_bSelected = bSelected;
    }
}

void ZenoBackgroundItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QPainterPath path = shape();
    if (m_img) {
        painter->setClipPath(path);
    } else {
        if (m_bSelected) {
            painter->fillPath(path, m_clrSelected);
        } else {
            painter->fillPath(path, m_clrNormal);
        }
    }
}


///////////////////////////////////////////////////////////////////////////
ZenoBackgroundWidget::ZenoBackgroundWidget(QGraphicsItem *parent, Qt::WindowFlags wFlags)
    : QGraphicsWidget(parent, wFlags)
    , lt_radius(0)
    , rt_radius(0)
    , lb_radius(0)
    , rb_radius(0)
    , m_borderWidth(0)
    , m_bFixRadius(true)
    , m_bSelected(false)
{
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    setAcceptHoverEvents(true);
}

QRectF ZenoBackgroundWidget::boundingRect() const
{
    QRectF rc = _base::boundingRect();
    qreal halfpw = (qreal)m_borderWidth / 2;
	//if (halfpw > 0.0)
 //       rc.adjust(-halfpw, -halfpw, halfpw, halfpw);
    return rc;
}

void ZenoBackgroundWidget::setBorder(qreal width, const QColor& clrBorder)
{
    m_borderWidth = width;
    m_clrBorder = clrBorder;
}

void ZenoBackgroundWidget::setColors(bool bAcceptHovers, const QColor &clrNormal, const QColor &clrHovered, const QColor &clrSelected)
{
    setAcceptHoverEvents(bAcceptHovers);
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    m_color = m_clrNormal;
    update();
}

void ZenoBackgroundWidget::setRadius(int lt, int rt, int lb, int rb)
{
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

QSizeF ZenoBackgroundWidget::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    return QGraphicsWidget::sizeHint(which, constraint);
}

void ZenoBackgroundWidget::setGeometry(const QRectF& rect)
{
    QGraphicsWidget::setGeometry(rect);
}

void ZenoBackgroundWidget::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
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

void ZenoBackgroundWidget::toggle(bool bSelected)
{
    m_bSelected = bSelected;
    if (m_clrSelected.isValid())
    {
        m_color = m_bSelected ? m_clrSelected : m_clrNormal;
        update();
    }
}

QPainterPath ZenoBackgroundWidget::shape() const
{
    QRectF r = boundingRect();
    r.adjust(m_borderWidth / 2, m_borderWidth / 2, -m_borderWidth / 2, -m_borderWidth / 2);
    return UiHelper::getRoundPath(r, lt_radius, rt_radius, lb_radius, rb_radius, m_bFixRadius);
}

void ZenoBackgroundWidget::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    _base::hoverEnterEvent(event);
    m_color = m_clrHovered;
    update();
}

void ZenoBackgroundWidget::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    _base::hoverMoveEvent(event);
}

void ZenoBackgroundWidget::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    _base::hoverLeaveEvent(event);
    if (m_clrSelected.isValid())
    {
		m_color = isSelected() ? m_clrSelected : m_clrNormal;
		update();
    }
}

void ZenoBackgroundWidget::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
}

void ZenoBackgroundWidget::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
}