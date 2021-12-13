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
ResizableRectItem::ResizableRectItem(const BackgroundComponent& comp, QGraphicsItem *parent)
    : m_rect(QRectF(0, 0, comp.rc.width(), comp.rc.height()))
    , lt_radius(comp.lt_radius)
    , rt_radius(comp.rt_radius)
    , lb_radius(comp.lb_radius)
    , rb_radius(comp.rb_radius)
    , m_bFixRadius(true)
    , m_img(nullptr)
{
    if (!comp.imageElem.image.isEmpty()) {
        m_img = new ZenoImageItem(comp.imageElem.image, comp.imageElem.imageHovered, comp.imageElem.imageOn, comp.rc.size(), this);
        m_img->setZValue(100);
        m_img->show();
    }
}

QRectF ResizableRectItem::boundingRect() const
{
	return m_rect;
}

void ResizableRectItem::resize(QSizeF sz)
{
    QPointF topLeft = m_rect.topLeft();
    m_rect = QRectF(topLeft.x(), topLeft.y(), sz.width(), sz.height());
    if (m_img)
        m_img->resize(sz);
}

void ResizableRectItem::setColors(const QColor& clrNormal, const QColor& clrHovered, const QColor& clrSelected)
{
    m_clrNormal = clrNormal;
    m_clrHovered = clrHovered;
    m_clrSelected = clrSelected;
    update();
}

void ResizableRectItem::setRadius(int lt, int rt, int lb, int rb)
{
    lt_radius = lt;
    rt_radius = rt;
    lb_radius = lb;
    rb_radius = rb;
    update();
}

std::pair<qreal, qreal> ResizableRectItem::getRxx2(QRectF r, qreal xRadius, qreal yRadius, bool AbsoluteSize) const
{
    if (AbsoluteSize) {
        qreal w = r.width() / 2;
        qreal h = r.height() / 2;

        if (w == 0) {
            xRadius = 0;
        } else {
            xRadius = 100 * qMin(xRadius, w) / w;
        }
        if (h == 0) {
            yRadius = 0;
        } else {
            yRadius = 100 * qMin(yRadius, h) / h;
        }
    } else {
        if (xRadius > 100)// fix ranges
            xRadius = 100;

        if (yRadius > 100)
            yRadius = 100;
    }

    qreal w = r.width();
    qreal h = r.height();
    qreal rxx2 = w * xRadius / 100;
    qreal ryy2 = h * yRadius / 100;
    return std::make_pair(rxx2, ryy2);
}

QPainterPath ResizableRectItem::shape() const
{
    QPainterPath path;
    QRectF r = m_rect.normalized();

    if (r.isNull())
        return path;

    if (lt_radius <= 0 && rt_radius <= 0 && lb_radius <= 0 && rb_radius <= 0)
    {
        path.addRect(r);
        return path;
    }

    qreal x = r.x();
    qreal y = r.y();
    qreal w = r.width();
    qreal h = r.height();

    auto pair = getRxx2(r, lt_radius, lt_radius, m_bFixRadius);
    qreal rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0)
    {
        path.moveTo(x, y);
    } else {
        path.arcMoveTo(x, y, rxx2, ryy2, 180);
        path.arcTo(x, y, rxx2, ryy2, 180, -90);
    }

    pair = getRxx2(r, rt_radius, rt_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y);
    } else {
        path.arcTo(x + w - rxx2, y, rxx2, ryy2, 90, -90);
    }

    pair = getRxx2(r, rb_radius, rb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x + w, y + h);
    } else {
        path.arcTo(x + w - rxx2, y + h - rxx2, rxx2, ryy2, 0, -90);
    }

    pair = getRxx2(r, lb_radius, lb_radius, m_bFixRadius);
    rxx2 = pair.first, ryy2 = pair.second;
    if (rxx2 <= 0) {
        path.lineTo(x, y + h);
    } else {
        path.arcTo(x, y + h - rxx2, rxx2, ryy2, 270, -90);
    }

    path.closeSubpath();
    return path;
}

void ResizableRectItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    QPainterPath path = shape();
    if (m_img) {
        painter->setClipPath(path);
        //m_img->paint(painter, option, widget);
    } else {
        painter->fillPath(path, m_clrNormal);
    }
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