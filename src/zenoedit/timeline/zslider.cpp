#include "zslider.h"

ZSlider::ZSlider(QWidget* parent)
    : QWidget(parent)
    , m_left(0.)
    , m_right(100.)
    , m_value(0.)
{
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

QSize ZSlider::sizeHint() const
{
    return QSize(0, 64);
}

void ZSlider::mousePressEvent(QMouseEvent* event)
{
    QWidget::mousePressEvent(event);
    QPoint pt = event->pos();
    setSliderValue(_posToFrame(pt.x()));
}

void ZSlider::mouseMoveEvent(QMouseEvent* event)
{
    QPoint pt = event->pos();
    setSliderValue(_posToFrame(pt.x()));
}

void drawText(QPainter* painter, qreal x, qreal y, Qt::Alignment flags,
    const QString& text, QRectF* boundingRect = 0)
{
    const qreal size = 32767.0;
    QPointF corner(x, y - size);
    if (flags & Qt::AlignHCenter) corner.rx() -= size / 2.0;
    else if (flags & Qt::AlignRight) corner.rx() -= size;
    if (flags & Qt::AlignVCenter) corner.ry() += size / 2.0;
    else if (flags & Qt::AlignTop) corner.ry() += size;
    else flags |= Qt::AlignBottom;
    QRectF rect{ corner.x(), corner.y(), size, size };
    painter->drawText(rect, flags, text, boundingRect);
}

void ZSlider::setSliderValue(int value)
{
    m_value = qMin(qMax(m_left, value), m_right);
    update();
    emit sliderValueChange(value);
}

int ZSlider::_posToFrame(int x)
{
    int W = width();
    qreal rate = (qreal)(x + 5 - m_sHMargin) / (W - 2 * m_sHMargin);
    return rate * (m_right - m_left + 1) + m_left;
}

int ZSlider::_frameToPos(int frame)
{
    qreal distPerFrame = (qreal)(width() - 2 * m_sHMargin) / (m_right - m_left + 1);
    return m_sHMargin + frame * distPerFrame;
}

int ZSlider::_getframes()
{
    int frames = 0;
    int n = m_right - m_left + 1;
    int W = width();
    if (W < 250)
    {
        frames = 1. / 15 * n;
    }
    else if (W < 500)
    {
        frames = 1. / 7 * n;
    }
    else if (W < 700)
    {
        frames = 1. / 2 * n;
    }
    else
    {
        frames = n;
    }
    return frames;
}

void ZSlider::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    int n = m_right - m_left + 1;
    int frames = _getframes();

    QFont font("Calibre", 9);
    QFontMetrics metrics(font);
    painter.setFont(font);
    painter.setPen(QPen(QColor(153, 153, 153)));

    for (int i = m_left, j = 0; i <= m_right; i += (n / frames), j++)
    {
        int h = 0, ytext = height() / 2 - h / 2;
        int x = _frameToPos(i);
        if (j % 10 == 0)
        {
            h = 30;

            //draw time tick
            int xpos = _frameToPos(i);
            QString scaleValue = QString::number(i);
            int textWidth = metrics.horizontalAdvance(scaleValue);
            painter.drawText(QPoint(xpos - textWidth / 2, height() / 2 - 20), scaleValue);
        }
        else
        {
            h = 15;
        }
        int y = height() / 2 - h / 2;
        painter.drawLine(QPointF(x, y), QPointF(x, y + h));
    }

    drawSlideHandle(&painter);
}

void ZSlider::drawSlideHandle(QPainter* painter)
{
    //draw time slider
    static const qreal handler_width = 26;
    static const qreal handler_height = 18;
    static const qreal handler_arrow_width = 10;
    static const qreal handler_arrow_height = 5;

    qreal xleftmost = _frameToPos(m_left);
    qreal xrightmost = _frameToPos(m_right);
    qreal xarrow_pos = _frameToPos(m_value);
    qreal yarrow_pos = height() / 2 - 15. / 2;

    qreal x_slider_left = 0, x_slider_right = 0;
    x_slider_left = xarrow_pos - handler_width / 2.;
    x_slider_right = x_slider_left + handler_width - 1;

    QPainterPath handlerPath;
    handlerPath.moveTo(QPointF(xarrow_pos, yarrow_pos));

    qreal x_arrow_right = qMin(xarrow_pos + handler_arrow_width / 2, x_slider_right);
    handlerPath.lineTo(QPointF(x_arrow_right, yarrow_pos - handler_arrow_height));
    handlerPath.lineTo(QPointF(x_slider_right, yarrow_pos - handler_arrow_height));
    handlerPath.lineTo(QPointF(x_slider_right, yarrow_pos - handler_arrow_height - handler_height));
    handlerPath.lineTo(QPointF(x_slider_left, yarrow_pos - handler_arrow_height - handler_height));
    handlerPath.lineTo(QPointF(x_slider_left, yarrow_pos - handler_arrow_height));

    qreal x_arrow_left = qMax(xarrow_pos - handler_arrow_width / 2, x_slider_left);
    handlerPath.lineTo(QPointF(x_arrow_left, yarrow_pos - handler_arrow_height));
    handlerPath.lineTo(QPointF(xarrow_pos, yarrow_pos));

    painter->fillPath(handlerPath, QColor(5, 5, 5));
    painter->drawPath(handlerPath);

    QString showText(QString::number(m_value));
    QFont font("Calibre", 9);
    QFontMetrics metrics(font);
    painter->setFont(font);
    painter->setPen(QPen(QColor(153, 153, 153)));
    int textWidth = metrics.horizontalAdvance(showText);
    int textHeight = metrics.height();
    drawText(painter, (x_slider_left + x_slider_right) / 2., yarrow_pos - handler_arrow_height - handler_height / 2, Qt::AlignVCenter | Qt::AlignHCenter, showText);
}
