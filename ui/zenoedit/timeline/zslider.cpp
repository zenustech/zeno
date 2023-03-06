#include "zslider.h"
#include <zenoui/style/zenostyle.h>


ZSlider::ZSlider(QWidget* parent)
    : QWidget(parent)
    , m_value(0)
    , m_from(0)
    , m_to(100)
{
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
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

static void drawText(QPainter* painter, qreal x, qreal y, Qt::Alignment flags,
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
    int newVal = qMin(qMax(m_from, value), m_to);
    if (newVal == m_value)
        return;
    m_value = newVal;
    update();
    emit sliderValueChange(m_value);
}

int ZSlider::_posToFrame(int x)
{
    int W = width();
    qreal rate = (qreal)(x + 5 - m_sHMargin) / (W - 2 * m_sHMargin);
    return rate * (m_to - m_from + 1) + m_from;
}

int ZSlider::_frameToPos(int frame)
{
    qreal W = width() - 2 * m_sHMargin;
    qreal distPerFrame = (qreal)(width() - 2 * m_sHMargin) / (m_to - m_from + 1);
    return m_sHMargin + (frame - m_from) * distPerFrame;
}

void ZSlider::setFromTo(int from, int to)
{
    m_from = from;
    m_to = to;
    if (m_value < m_from)
    {
        m_value = m_from;
        emit sliderValueChange(m_value);
    }
    else if (m_value > m_to)
    {
        m_value = m_to;
        emit sliderValueChange(m_value);
    }
    update();
}

int ZSlider::value() const
{
    return m_value;
}

int ZSlider::_getframes()
{
    //todo
    int frames = 0;
    int n = m_to - m_from + 1;
    int W = width();
    if (W < 250)
    {
        frames = 50;
    }
    else if (W < 500)
    {
        frames = 100;
    }
    else if (W < 700)
    {
        frames = 100;
    }
    else
    {
        frames = 100;
    }
    return frames;
}

QSize ZSlider::sizeHint() const
{
    int h = ZenoStyle::dpiScaled(scaleH + fontHeight + fontScaleSpacing);
    return QSize(0, h);
}

void ZSlider::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    int n = m_to - m_from + 1;
    int frames = _getframes();

    QFont font("Segoe UI", 9);
    font.setWeight(QFont::DemiBold);
    QFontMetrics metrics(font);
    int hh = metrics.height();

    painter.setFont(font);

    for (int i = m_from; i <= m_to; i++)
    {
        int x = _frameToPos(i);
        QString scaleValue = QString::number(i);
        painter.setPen(QPen(QColor("#5A646F"), 1));

        if (i % 5 == 0)
        {
            //draw time tick
            int h = ZenoStyle::dpiScaled(scaleH);
            int xpos = _frameToPos(i);
            int textWidth = metrics.horizontalAdvance(scaleValue);
            //don't know the y value.
            int yText = height() * 0.4;
            if (m_value != i)
                painter.drawText(QPoint(xpos - textWidth / 2, yText), scaleValue);

            int y = height() - h;
            painter.drawLine(QPointF(x, y), QPointF(x, y + h));
        }
        else
        {
            int h = ZenoStyle::dpiScaled(smallScaleH);
            int y = height() - h;
            painter.drawLine(QPointF(x, y), QPointF(x, y + h));
        }
    }

    painter.setPen(QPen(QColor("#335A646F"), 1));
    //painter.drawLine(QPointF(_frameToPos(m_from), height() - 5 - 4), QPointF(_frameToPos(m_to), height() - 5 - 4));
    drawSlideHandle(&painter, scaleH);
}

void ZSlider::drawSlideHandle(QPainter* painter, int scaleH)
{
    //draw time slider
    qreal xleftmost = _frameToPos(m_from);
    qreal xrightmost = _frameToPos(m_to);
    qreal xarrow_pos = _frameToPos(m_value);

    painter->setPen(Qt::NoPen);
    int y = height() - scaleH;
    painter->fillRect(QRectF(QPointF(xleftmost, y), QPointF(xarrow_pos, y + scaleH)), QColor(76, 159, 244, 64));

    //draw handle.
    static const int handleHeight = ZenoStyle::dpiScaled(12);
    static const int handleWidth = ZenoStyle::dpiScaled(6);
    y = height() - handleHeight;
    qreal x = xarrow_pos - handleWidth / 2;
    painter->fillRect(QRectF(QPointF(xarrow_pos - handleWidth / 2, y),
                             QPointF(xarrow_pos + handleWidth / 2, y + handleHeight)),
                      QColor(76, 159, 244));

    QFont font("Segoe UI Bold", 10);
    QFontMetrics metrics(font);
    painter->setFont(font);

    QString numText = QString::number(m_value);
    int w = metrics.horizontalAdvance(numText);
    painter->setPen(QColor(76, 159, 244));
    painter->drawText(QPointF(xarrow_pos - w / 2, height() * 0.4), numText);
}
