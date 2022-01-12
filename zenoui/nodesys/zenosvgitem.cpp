#include "zenosvgitem.h"
#include <QSvgRenderer>


ZenoSvgItem::ZenoSvgItem(QGraphicsItem *parent)
    : QGraphicsSvgItem(parent)
    , m_size(-1.0, -1.0)
{
}

ZenoSvgItem::ZenoSvgItem(const QString &image, QGraphicsItem *parent)
    : QGraphicsSvgItem(image, parent)
    , m_size(-1.0, -1.0)
{
}

void ZenoSvgItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(widget);
    Q_UNUSED(option);
    if (!renderer()->isValid())
        return;

    if (elementId().isEmpty())
        renderer()->render(painter, boundingRect());
    else
        renderer()->render(painter, elementId(), boundingRect());
}

void ZenoSvgItem::setSize(QSizeF size)
{
    if (m_size != size)
    {
        m_size = size;
        update();
    }
}

QRectF ZenoSvgItem::boundingRect() const
{
    return QRectF(QPointF(0.0, 0.0), m_size);
}


ZenoImageItem::ZenoImageItem(const ImageElement& elem, const QSizeF& sz, QGraphicsItem* parent)
    : m_svg(nullptr)
    , _base(parent)
    , m_normal(elem.image)
    , m_hovered(elem.imageHovered)
    , m_selected(elem.imageOn)
    , m_size(sz)
    , m_bToggled(false)
{
    setAcceptHoverEvents(true);
    m_svg = new ZenoSvgItem(m_normal, this);
    m_svg->setSize(m_size);
}

ZenoImageItem::ZenoImageItem(const QString &normal, const QString &hovered, const QString &selected, const QSizeF &sz, QGraphicsItem *parent)
    : m_svg(nullptr)
    , _base(parent)
    , m_normal(normal)
    , m_hovered(hovered)
    , m_selected(selected)
    , m_size(sz)
    , m_bToggled(false)
{
    setAcceptHoverEvents(true);
    m_svg = new ZenoSvgItem(m_normal, this);
    m_svg->setSize(m_size);
}

QRectF ZenoImageItem::boundingRect() const
{
    return m_svg->boundingRect();
}

void ZenoImageItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}

void ZenoImageItem::resize(QSizeF sz)
{
    m_size = sz;
    m_svg->setSize(m_size);
}

void ZenoImageItem::toggle(bool bToggled)
{
    m_bToggled = bToggled;
    if (m_bToggled) {
        delete m_svg;
        m_svg = new ZenoSvgItem(m_selected, this);
        m_svg->setSize(m_size);
    } else {
        delete m_svg;
        m_svg = new ZenoSvgItem(m_normal, this);
        m_svg->setSize(m_size);
    }
}

void ZenoImageItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    _base::mousePressEvent(event);
    emit clicked();
}

void ZenoImageItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    if (!m_bToggled && !m_hovered.isEmpty())
    {
        delete m_svg;
        m_svg = new ZenoSvgItem(m_hovered, this);
        m_svg->setSize(m_size);
    }
    _base::hoverEnterEvent(event);
}

void ZenoImageItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoImageItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    if (!m_bToggled && !m_hovered.isEmpty())
    {
        delete m_svg;
        m_svg = new ZenoSvgItem(m_normal, this);
        m_svg->setSize(m_size);
    }
    _base::hoverLeaveEvent(event);
}