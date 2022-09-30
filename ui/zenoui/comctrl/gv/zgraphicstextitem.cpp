#include "zgraphicstextitem.h"


ZGraphicsTextItem::ZGraphicsTextItem(const QString& text, const QFont& font, const QColor& color, QGraphicsItem* parent)
    : QGraphicsTextItem(parent)
    , m_text(text)
{

}

void ZGraphicsTextItem::setText(const QString& text)
{
    m_text = text;
    setPlainText(m_text);
}

void ZGraphicsTextItem::setMargins(qreal leftM, qreal topM, qreal rightM, qreal bottomM)
{

}

void ZGraphicsTextItem::setBackground(const QColor& clr)
{

}

QRectF ZGraphicsTextItem::boundingRect() const
{
    //todo
    return QRectF();
}

void ZGraphicsTextItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{

}

QPainterPath ZGraphicsTextItem::shape() const
{
    //todo
    QPainterPath path;
    return path;
}

void ZGraphicsTextItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mousePressEvent(event);
}

void ZGraphicsTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mouseMoveEvent(event);
}

void ZGraphicsTextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsTextItem::mouseReleaseEvent(event);
}



ZSimpleTextItem::ZSimpleTextItem(QGraphicsItem* parent)
    : base(parent)
    , m_fixedWidth(-1)
    , m_bRight(false)
    , m_bHovered(false)
    , m_alignment(Qt::AlignLeft)
    , m_hoverCursor(Qt::ArrowCursor)
{
    setFlags(ItemIsFocusable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZSimpleTextItem::ZSimpleTextItem(const QString& text, QGraphicsItem* parent)
    : base(text, parent)
    , m_fixedWidth(-1)
    , m_bRight(false)
    , m_bHovered(false)
    , m_alignment(Qt::AlignLeft)
{
    setAcceptHoverEvents(true);
    updateBoundingRect();
}

ZSimpleTextItem::~ZSimpleTextItem()
{

}

QRectF ZSimpleTextItem::boundingRect() const
{
    return m_boundingRect;
}

QPainterPath ZSimpleTextItem::shape() const
{
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZSimpleTextItem::updateBoundingRect()
{
    QTextLayout layout(text(), font());
    QSizeF sz = size(text(), font(), m_padding.left, m_padding.top, m_padding.right, m_padding.bottom);
    if (m_fixedWidth > 0)
    {
        m_boundingRect = QRectF(0, 0, m_fixedWidth, sz.height());
    }
    else
    {
        m_boundingRect = QRectF(0, 0, sz.width(), sz.height());
    }
}

void ZSimpleTextItem::setPadding(int left, int top, int right, int bottom)
{
    m_padding.left = left;
    m_padding.right = right;
    m_padding.top = top;
    m_padding.bottom = bottom;
    updateBoundingRect();
}

void ZSimpleTextItem::setAlignment(Qt::Alignment align)
{
    m_alignment = align;
}

void ZSimpleTextItem::setFixedWidth(qreal fixedWidth)
{
    m_fixedWidth = fixedWidth;
    updateBoundingRect();
}

QRectF ZSimpleTextItem::setupTextLayout(QTextLayout* layout, _padding padding, Qt::Alignment align, qreal fixedWidth)
{
    layout->setCacheEnabled(true);
    layout->beginLayout();
    while (layout->createLine().isValid())
        ;
    layout->endLayout();
    qreal maxWidth = 0;
    qreal y = 0;
    for (int i = 0; i < layout->lineCount(); ++i) {
        QTextLine line = layout->lineAt(i);
        qreal wtf = line.width();
        maxWidth = qMax(maxWidth, line.naturalTextWidth() + padding.left + padding.right);

        qreal x = 0;
        qreal w = line.horizontalAdvance();
        if (fixedWidth > 0)
        {
            if (align == Qt::AlignCenter)
            {
                x = (fixedWidth - w) / 2;
            }
            else if (align == Qt::AlignRight)
            {
                x = (fixedWidth - w);
            }
        }
        line.setPosition(QPointF(x, y + padding.top));
        y += line.height() + padding.top + padding.bottom;
    }
    return QRectF(0, 0, maxWidth, y);
}

void ZSimpleTextItem::setBackground(const QColor& clr)
{
    m_bg = clr;
}

void ZSimpleTextItem::setHoverCursor(Qt::CursorShape cursor)
{
    m_hoverCursor = cursor;
}

void ZSimpleTextItem::setRight(bool right)
{
    m_bRight = right;
}

bool ZSimpleTextItem::isHovered() const
{
    return m_bHovered;
}

QSizeF ZSimpleTextItem::size(const QString& text, const QFont& font, int pleft, int pTop, int pRight, int pBottom)
{
    QTextLayout layout(text, font);
    QRectF rc = setupTextLayout(&layout, _padding(pleft, pTop, pRight, pBottom));
    return rc.size();
}

void ZSimpleTextItem::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    m_bHovered = true;
    base::hoverEnterEvent(event);
    setCursor(m_hoverCursor);
}

void ZSimpleTextItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    base::hoverMoveEvent(event);
}

void ZSimpleTextItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    base::hoverLeaveEvent(event);
    m_bHovered = false;
    setCursor(Qt::ArrowCursor);
}

void ZSimpleTextItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    base::mousePressEvent(event);
}

void ZSimpleTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    base::mouseMoveEvent(event);
}

void ZSimpleTextItem::keyPressEvent(QKeyEvent* event)
{
    base::keyPressEvent(event);
}

void ZSimpleTextItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (m_bg.isValid())
    {
        painter->save();
        painter->setPen(Qt::NoPen);
        painter->setBrush(m_bg);
        painter->drawRect(boundingRect());
        painter->restore();
    }

    painter->setFont(this->font());

    QString tmp = text();
    tmp.replace(QLatin1Char('\n'), QChar::LineSeparator);
    QTextLayout layout(tmp, font());

    QPen p;
    if (option->state & QStyle::State_MouseOver)
    {
        p.setBrush(QColor(255,255,255));
    }
    else
    {
        p.setBrush(brush());
    }

    painter->setPen(p);
    if (pen().style() == Qt::NoPen && brush().style() == Qt::SolidPattern) {
        painter->setBrush(Qt::NoBrush);
    }
    else {
        QTextLayout::FormatRange range;
        range.start = 0;
        range.length = layout.text().length();
        range.format.setTextOutline(pen());
        layout.setFormats(QVector<QTextLayout::FormatRange>(1, range));
    }

    qreal w = boundingRect().width();
    setupTextLayout(&layout, m_padding, m_alignment, m_fixedWidth == -1 ? w : m_fixedWidth);

    layout.draw(painter, QPointF(0, 0));
}


ZSimpleTextLayoutItem::ZSimpleTextLayoutItem(const QString& text, QGraphicsItem* parent)
    : QGraphicsLayoutItem()
    , ZSimpleTextItem(text, parent)
{
    setZValue(3);
    setGraphicsItem(this);
    setFlags(ItemSendsScenePositionChanges);
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
}

void ZSimpleTextLayoutItem::setGeometry(const QRectF& rect)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(rect);
    setPos(rect.topLeft());
}

QRectF ZSimpleTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

QPainterPath ZSimpleTextLayoutItem::shape() const
{
    QPainterPath path;
    path.addRect(boundingRect());
    return path;
}

void ZSimpleTextLayoutItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    //painter->fillRect(boundingRect(), QColor(255, 0, 0));
    ZSimpleTextItem::paint(painter, option, widget);
}

QSizeF ZSimpleTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    QRectF rc = ZSimpleTextItem::boundingRect();
    switch (which)
    {
    case Qt::MinimumSize:
    case Qt::PreferredSize:
        return rc.size();
    case Qt::MaximumSize:
        return QSizeF(3000, rc.height());
    default:
        break;
    }
    return constraint;
}
