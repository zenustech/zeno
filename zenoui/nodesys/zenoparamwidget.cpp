#include "zenoparamwidget.h"
#include "zenosocketitem.h"
#include "../render/common_id.h"


ZenoParamWidget::ZenoParamWidget(QGraphicsItem* parent, Qt::WindowFlags wFlags)
    : QGraphicsProxyWidget(parent, wFlags)
{
}

ZenoParamWidget::~ZenoParamWidget()
{
}

int ZenoParamWidget::type() const
{
    return Type;
}

void ZenoParamWidget::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    QSizeF sz = this->sizeHint(Qt::PreferredSize);
    sz = sizeHint(Qt::MinimumSize);
    sz = sizeHint(Qt::MaximumSize);
    QGraphicsProxyWidget::paint(painter, option, widget);
}


ZenoParamLineEdit::ZenoParamLineEdit(const QString &text, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_pLineEdit = new QLineEdit;
    m_pLineEdit->setText(text);
    
    setWidget(m_pLineEdit);
}

ZenoParamLabel::ZenoParamLabel(const QString &text, const QFont &font, const QBrush &fill, QGraphicsItem *parent)
    : ZenoParamWidget(parent)
{
    m_label = new QLabel(text);

    QPalette palette;
    palette.setColor(QPalette::WindowText, fill.color());
    m_label->setFont(font);
    m_label->setPalette(palette);
    m_label->setAttribute(Qt::WA_TranslucentBackground);
    m_label->setAutoFillBackground(true);
    setWidget(m_label);
}

void ZenoParamLabel::setAlignment(Qt::Alignment alignment)
{
    m_label->setAlignment(alignment);
}


ZenoTextLayoutItem::ZenoTextLayoutItem(const QString &text, const QFont &font, const QColor &color, QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsTextItem(text, parent)
    , m_text(text)
{
    setZValue(ZVALUE_ELEMENT);
    setFont(font);
    setDefaultTextColor(color);
    
    setGraphicsItem(this);
    setFlags(ItemSendsScenePositionChanges);
}

void ZenoTextLayoutItem::setGeometry(const QRectF& geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
    //emit geometrySetup(scenePos());
}

QRectF ZenoTextLayoutItem::boundingRect() const
{
    QRectF rc = QRectF(QPointF(0, 0), geometry().size());
    return rc;
}

void ZenoTextLayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    //painter->setBrush(Qt::black);
    //painter->drawRect(boundingRect());  //FOR DEBUG LAOUT POSITION
    QGraphicsTextItem::paint(painter, option, widget);
}

QSizeF ZenoTextLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which) {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            return QGraphicsTextItem::boundingRect().size();
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
        default:
            break;
    }
    return constraint;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
ZenoSvgLayoutItem::ZenoSvgLayoutItem(ZenoSocketItem* item, QGraphicsLayoutItem* parent, bool isLayout)
    : QGraphicsLayoutItem(parent, isLayout)
    , m_item(item)
{
    setGraphicsItem(m_item);
}

QSizeF ZenoSvgLayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    QSizeF sz = m_item->size();
    switch (which) {
        case Qt::MinimumSize:
            return sz;
        case Qt::PreferredSize:
            return sz;
        case Qt::MaximumSize:
            return sz;
        default:
            //return m_rect.size();
            return sz;
    }
}

void ZenoSvgLayoutItem::setGeometry(const QRectF &rect)
{
    if (m_item)
        m_item->setPos(rect.topLeft());
    QGraphicsLayoutItem::setGeometry(rect);
}

void ZenoSvgLayoutItem::updateGeometry()
{
    QGraphicsLayoutItem::updateGeometry();
}


//////////////////////////////////////////////////////////////////////
LayoutItem::LayoutItem(QGraphicsItem *parent)
    : QGraphicsLayoutItem()
    , QGraphicsItem(parent)
    , m_pix(QPixmap(QLatin1String(":/icons/checked.png")))
{
    setGraphicsItem(this);
}

void LayoutItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(widget);
    Q_UNUSED(option);

    QRectF frame(QPointF(0, 0), geometry().size());
    const QSize pmSize = m_pix.size();
    QGradientStops stops;

    // paint a background rect (with gradient)
    QLinearGradient gradient(frame.topLeft(), frame.topLeft() + QPointF(200, 200));
    stops << QGradientStop(0.0, QColor(60, 60, 60));
    stops << QGradientStop(frame.height() / 2 / frame.height(), QColor(102, 176, 54));

    //stops << QGradientStop(((frame.height() + h)/2 )/frame.height(), QColor(157, 195,  55));
    stops << QGradientStop(1.0, QColor(215, 215, 215));
    gradient.setStops(stops);
    painter->setBrush(QBrush(gradient));
    painter->drawRoundedRect(frame, 10.0, 10.0);

    // paint a rect around the pixmap (with gradient)
    QPointF pixpos = frame.center() - (QPointF(pmSize.width(), pmSize.height()) / 2);
    QRectF innerFrame(pixpos, pmSize);
    innerFrame.adjust(-4, -4, 4, 4);
    gradient.setStart(innerFrame.topLeft());
    gradient.setFinalStop(innerFrame.bottomRight());
    stops.clear();
    stops << QGradientStop(0.0, QColor(215, 255, 200));
    stops << QGradientStop(0.5, QColor(102, 176, 54));
    stops << QGradientStop(1.0, QColor(0, 0, 0));
    gradient.setStops(stops);
    painter->setBrush(QBrush(gradient));
    painter->drawRoundedRect(innerFrame, 10.0, 10.0);
    painter->drawPixmap(pixpos, m_pix);
}

QRectF LayoutItem::boundingRect() const
{
    return QRectF(QPointF(0, 0), geometry().size());
}

void LayoutItem::setGeometry(const QRectF &geom)
{
    prepareGeometryChange();
    QGraphicsLayoutItem::setGeometry(geom);
    setPos(geom.topLeft());
}

QSizeF LayoutItem::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    switch (which) {
        case Qt::MinimumSize:
        case Qt::PreferredSize:
            // Do not allow a size smaller than the pixmap with two frames around it.
            return m_pix.size() + QSize(12, 12);
        case Qt::MaximumSize:
            return QSizeF(1000, 1000);
        default:
            break;
    }
    return constraint;
}