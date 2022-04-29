#include "zslider.h"
#include <zenoui/style/zenostyle.h>


ZSlider::ZSlider(QWidget* parent)
    : QWidget(parent)
    , m_left(0)
    , m_right(100)
    , m_value(0)
    , m_from(m_left)
    , m_to(m_right)
{
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

QSize ZSlider::sizeHint() const
{
    return ZenoStyle::dpiScaledSize(QSize(0, 45));
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
    m_value = qMin(qMax(m_from, value), m_to - 1);
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

void ZSlider::setFromTo(int from, int to)
{
    m_from = from;
    m_to = to;
    m_value = m_from;
    update();
}

int ZSlider::_getframes()
{
    int frames = 0;
    int n = m_right - m_left + 1;
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

void ZSlider::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    int n = m_right - m_left + 1;
    int frames = _getframes();

    QFont font("HarmonyOS Sans", 10);
    QFontMetrics metrics(font);
    painter.setFont(font);

    for (int i = m_left, j = 0; i <= m_right; i += (n / std::max(1, frames)), j++)
    {
        int h = 0;
        int x = _frameToPos(i);
        QString scaleValue = QString::number(i);
        painter.setPen(QPen(QColor(119, 119, 119), 2));

        if (j % 5 == 0)
        {
            h = 16;

            //draw time tick
            int xpos = _frameToPos(i);
            int textWidth = metrics.horizontalAdvance(scaleValue);
            //don't know the y value.
            int yText = height() * 0.4;
            if (m_value != i)
                painter.drawText(QPoint(xpos - textWidth / 2, yText), scaleValue);

            int y = height() - h - 2;
            painter.drawLine(QPointF(x, y), QPointF(x, y + h));
        }
        else
        {
            h = 8;
            int y = height() - h - 2;
            painter.drawLine(QPointF(x, y), QPointF(x, y + h - 3));
        }
    }

    painter.setPen(QPen(QColor(58, 58, 58), 2));
    painter.drawLine(QPointF(_frameToPos(m_left), height() - 2), QPointF(_frameToPos(m_right), height() - 2));
    drawSlideHandle(&painter);
}

void ZSlider::drawSlideHandle(QPainter* painter)
{
    //draw time slider
    qreal xleftmost = _frameToPos(m_left);
    qreal xrightmost = _frameToPos(m_right);
    qreal xarrow_pos = _frameToPos(m_value);
    qreal yarrow_pos = height() / 2 - 15. / 2;

    painter->setPen(Qt::NoPen);
    int y = height() - 10;
    painter->fillRect(QRectF(QPointF(xleftmost, y), QPointF(xarrow_pos, height() - 2)), QColor(76, 159, 244, 128));

    //draw handle.
    y = height() - 20;
    int w = 8;
    qreal x = xarrow_pos - w / 2;
    painter->fillRect(QRectF(QPointF(xarrow_pos - w / 2, y), QPointF(xarrow_pos + w / 2, height())), QColor(76, 159, 244));

    QFont font("HarmonyOS Sans", 15);
    font.setBold(true);
    QFontMetrics metrics(font);
    painter->setFont(font);

    QString numText = QString::number(m_value);
    w = metrics.horizontalAdvance(numText);
    painter->setPen(QColor(76, 159, 244));
    painter->drawText(QPointF(xarrow_pos - w / 2, height() * 0.4), numText);
}
