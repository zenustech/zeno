#include "resizecoreitem.h"
#include <QSvgRenderer>
#include <QStyleOptionGraphicsItem>

ResizableCoreItem::ResizableCoreItem(QGraphicsItem* parent)
	: QGraphicsItem(parent)
{

}

void ResizableCoreItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}


MySvgItem::MySvgItem(QGraphicsItem *parent)
    : QGraphicsSvgItem(parent), m_size(-1.0, -1.0)
{
}

MySvgItem::MySvgItem(const QString &fileName, QGraphicsItem *parent)
    : QGraphicsSvgItem(fileName, parent), m_size(-1.0, -1.0)
{
}

void MySvgItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
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

void MySvgItem::setSize(QSizeF size)
{
    if (m_size != size)
    {
        m_size = size;
        update();
    }
}

QRectF MySvgItem::boundingRect()
{
    return QRectF(QPointF(0.0, 0.0), m_size);
}


/////////////////////////////////////////////////////////////////////////////////////
ResizableImageItem::ResizableImageItem(const QString &normal, const QString &hovered, const QString &selected, QSizeF sz, QGraphicsItem *parent)
    : ResizableCoreItem(parent)
    , m_pixmap(nullptr)
    , m_svg(nullptr)
{
    this->setAcceptHoverEvents(true);
    resetImage(normal, hovered, selected, sz);
}

QRectF ResizableImageItem::boundingRect() const
{
    if (m_pixmap)
        return m_pixmap->boundingRect();
    else
        return m_svg->boundingRect();
}

bool ResizableImageItem::resetImage(const QString &normal, const QString &hovered, const QString &selected, QSizeF sz)
{
	QString suffix = QFileInfo(normal).completeSuffix();
	if (suffix.compare("svg", Qt::CaseInsensitive) == 0)
	{
        QFileInfo fileInfo(normal);
        QString name = fileInfo.fileName();
        QString ext = fileInfo.completeSuffix();

        if (m_pixmap) {
            delete m_pixmap;
            m_pixmap = nullptr;
        }

        if (m_svg) {
            delete m_svg;
            m_svg = nullptr;
        }

		m_svg = new MySvgItem(normal, this);
        m_svg->setZValue(ZVALUE_CORE_ITEM);
        m_svgNormal = normal;
        m_svgHovered = hovered;
        m_svgSelected = selected;
        m_size = sz;
        m_svg->setSize(m_size);
    }
	else
	{
        QFileInfo fileInfo(normal);
        QString name = fileInfo.fileName();
        QString ext = fileInfo.completeSuffix();

        if (m_svg)
            delete m_svg;
        m_svg = nullptr;

        m_normal = QPixmap(normal);
        m_hovered = QPixmap(hovered);
        m_selected = QPixmap(selected);
        m_pixmap = new QGraphicsPixmapItem(m_normal.scaled(sz.toSize()), this);
        m_pixmap->setZValue(ZVALUE_CORE_ITEM);
        m_size = sz;
    }
    return true;
}

void ResizableImageItem::resize(QSizeF sz)
{
    m_size = sz;
    if (m_pixmap) {
        m_pixmap->setPixmap(m_normal.scaled(m_size.toSize()));
    } else if (m_svg) {
        m_svg->setSize(m_size);
    }
}

void ResizableImageItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
    if (m_pixmap)
    {
        if (!m_hovered.isNull())
        {
            m_pixmap->setPixmap(m_hovered.scaled(m_size.toSize()));
        }
    } 
    else if (m_svg)
    {
        if (!m_svgHovered.isEmpty())
        {
            delete m_svg;
            m_svg = new MySvgItem(m_svgHovered, this);
            m_svg->setSize(m_size);
        }
    }
    _base::hoverEnterEvent(event);
}

void ResizableImageItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
    _base::hoverMoveEvent(event);
}

void ResizableImageItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event)
{
    if (m_pixmap) {
        m_pixmap->setPixmap(m_normal.scaled(m_size.toSize()));
    } else if (m_svg) {
        //todo
        delete m_svg;
        m_svg = new MySvgItem(m_svgNormal, this);
        m_svg->setSize(m_size);
    }
    _base::hoverLeaveEvent(event);
}


//////////////////////////////////////////////////////////////////////////////////////
ResizableRectItem::ResizableRectItem(QRectF rc, QGraphicsItem* parent)
	: m_rectItem(new QGraphicsRectItem(rc, parent))
{
	QPen pen(QColor(255, 0, 0), 1);
	pen.setJoinStyle(Qt::MiterJoin);
	m_rectItem->setPen(pen);
	m_rectItem->setBrush(QColor(142, 101, 101));
}

QRectF ResizableRectItem::boundingRect() const
{
	return m_rectItem->boundingRect();
}

void ResizableRectItem::resize(QSizeF sz)
{
	QPointF topLeft = m_rectItem->rect().topLeft();
	m_rectItem->setRect(QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height()));
}


///////////////////////////////////////////////////////////////////////////////////////
ResizableEclipseItem::ResizableEclipseItem(const QRectF& rect, QGraphicsItem* parent)
	: ResizableCoreItem(parent)
	, m_ellipseItem(new QGraphicsEllipseItem(rect, this))
{
	QPen pen(QColor(255, 0, 0), 1);
	pen.setJoinStyle(Qt::MiterJoin);
	m_ellipseItem->setPen(pen);
	m_ellipseItem->setBrush(QColor(142, 101, 101));
}

QRectF ResizableEclipseItem::boundingRect() const
{
	return m_ellipseItem->boundingRect();
}

void ResizableEclipseItem::resize(QSizeF sz)
{
	QPointF topLeft = m_ellipseItem->rect().topLeft();
	m_ellipseItem->setRect(QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height()));
}


////////////////////////////////////////////////////////////////////////////////////////
ResizableTextItem::ResizableTextItem(const QString& text, QGraphicsItem* parent)
	: m_pTextItem(new QGraphicsTextItem(text, this))
{
}

QRectF ResizableTextItem::boundingRect() const
{
	return m_pTextItem->boundingRect();
}

void ResizableTextItem::resize(QSizeF sz)
{
}

void ResizableTextItem::setText(const QString& text)
{
    m_pTextItem->setPlainText(text);
}

void ResizableTextItem::setTextProp(QFont font, QColor color)
{
    m_pTextItem->setFont(font);
    m_pTextItem->setDefaultTextColor(color);
}