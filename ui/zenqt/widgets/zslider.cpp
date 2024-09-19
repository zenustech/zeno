#include "zslider.h"
#include "style/zenostyle.h"
#include "zenoapplication.h"
#include <zeno/core/Session.h>
#include <zeno/extra/GlobalComm.h>
#include "zassert.h"


ZSlider::ZSlider(QWidget* parent)
    : QWidget(parent)
    , m_value(0)
    , m_from(0)
    , m_to(0)
    , m_cellLength(0)
    , m_sHMargin(ZenoStyle::dpiScaled(15))
    , scaleH(ZenoStyle::dpiScaled(9))
    , smallScaleH(ZenoStyle::dpiScaled(5))
    , fontHeight(ZenoStyle::dpiScaled(15))
    , fontScaleSpacing(ZenoStyle::dpiScaled(4))
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
    qreal distPerFrame = (qreal)(width() - 2 * m_sHMargin) / ((m_to - m_from) == 0 ? 1 : (m_to - m_from));
    qreal rate = (qreal)(x + distPerFrame / 2 - m_sHMargin) / (W - 2 * m_sHMargin);
    return rate * (m_to - m_from) + m_from;
}

int ZSlider::_frameToPos(int frame)
{
    qreal W = width() - 2 * m_sHMargin;
    qreal distPerFrame = (qreal)(width() - 2 * m_sHMargin) / ((m_to - m_from) == 0 ? 1 : (m_to - m_from));
    return m_sHMargin + (frame ) * distPerFrame;
}

void ZSlider::setFromTo(int from, int to)
{
    if (m_from == from && m_to == to)
        return;

    ZASSERT_EXIT(from <= to);

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
    m_cellLength = getCellLength(m_to - m_from);
    update();
}

int ZSlider::value() const
{
    return m_value;
}

void ZSlider::updateKeyFrames(const QVector<int> &keys) 
{
    m_keyframes = keys;
    update();
}

void ZSlider::updateDopnetworkFrameCached(int frame)
{
    m_dopNetworkframes.insert(frame);
    repaint();
}

void ZSlider::updateDopnetworkFrameRemoved(int frame)
{
    m_dopNetworkframes.remove(frame);
    repaint();
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

    QFont font = QApplication::font();
    font.setPointSize(9);
    font.setWeight(QFont::DemiBold);
    QFontMetrics metrics(font);
    painter.setFont(font);
    painter.setPen(QPen(QColor("#5A646F"), 1));
    painter.setRenderHint(QPainter::Antialiasing, false);

    int cellNum = ((m_to - m_from) / m_cellLength) + 1;
    int cellPixelLength = width() / cellNum;
    if (!(150 < cellPixelLength && cellPixelLength < 250)) {
        m_cellLength = getCellLength(m_to - m_from);
    } 
    int flag;
    for (int i = 2; i > -1; i--) {
        if (m_cellLength % m_lengthUnit[i] == 0) {
            m_cellLength / m_lengthUnit[i];
            flag = m_lengthUnit[i];
            break;
        }
    }
    int frameNum;
    int smallCellLength = m_cellLength / flag;
    int offset = 0;
    if (m_from % m_cellLength == 0) {
        //frameNum = ((m_to - m_from) / m_cellLength) * m_lengthUnit[i] + m_cellLength;
        frameNum = (m_to - m_from) / smallCellLength;
    } else {
        int firstCell = m_cellLength - m_from % m_cellLength + m_from;
        int smallCellNum = (firstCell - m_from) / smallCellLength;
        int h = ZenoStyle::dpiScaled(smallScaleH);
        int y = height() - h;
        for (int i = smallCellNum; i > 0; i--)
        {
            int x = _frameToPos(firstCell - smallCellLength * i - m_from);
            painter.drawLine(QPointF(x, y), QPointF(x, y + h));
        }
        offset = firstCell - m_from;
        frameNum = (m_to - m_from) / smallCellLength - smallCellNum - 1;
    }

    for (int i = 0; i <= frameNum; i++)
    {
        int cellScaleValue = (i / flag) * m_cellLength + offset;
        QString scaleValue = QString::number(cellScaleValue);
        int x = _frameToPos(smallCellLength * i + offset);

        if (i % flag == 0)
        {
            //draw time tick
            int h = ZenoStyle::dpiScaled(scaleH);
            int xpos = _frameToPos(scaleValue.toInt());
            int textWidth = metrics.horizontalAdvance(scaleValue);
            //don't know the y value.
            int yText = height() * 0.4;
            if (m_value != (cellScaleValue + m_from))
                painter.drawText(QPoint(xpos - textWidth / 2, yText), QString::number(cellScaleValue + m_from));

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

    painter.setPen(QPen(QColor("#5A646F"), 1));

    int left = _frameToPos(0);
    int right = width() - m_sHMargin;
    painter.drawLine(QPointF(left, height() - 1), QPointF(right, height() - 1));
    drawSlideHandle(&painter, scaleH);

    //draw dopnetwork cached frames
    for (auto frame : m_dopNetworkframes) {
        int x = _frameToPos(frame - m_from) + painter.pen().width();
        int h = ZenoStyle::dpiScaled(scaleH);
        int y = height() - h;
        int w = (qreal)(width() - 2 * m_sHMargin) / ((m_to - m_from) == 0 ? 1 : (m_to - m_from));
        QRect rec(x, y, w, h);
        painter.fillRect(rec, QColor("#8A2BE2"));
    }

    //draw keyframes
    for (auto frame : m_keyframes) {
        int x = _frameToPos(frame - m_from) + painter.pen().width();
        int h = ZenoStyle::dpiScaled(scaleH);
        int y = height() - h;
        int w = (qreal)(width() - 2 * m_sHMargin) / ((m_to - m_from) == 0 ? 1 : (m_to - m_from));
        QRect rec(x, y, w, h);
        painter.fillRect(rec, QColor("#3A6E64"));
    }

    //draw finished frame
    //return;
    auto& pGlobalComm = zeno::getSession().globalComm;
    int lastStateFrame = -1;
    zeno::GlobalComm::FRAME_STATE lastState = zeno::GlobalComm::FRAME_UNFINISH;
    for (int frame = m_from; frame <= m_to; frame++)
    {
        if (frame == m_from) {
            lastState = pGlobalComm->getFrameState(frame);
            lastStateFrame = m_from;
        }

        zeno::GlobalComm::FRAME_STATE currState = pGlobalComm->getFrameState(frame);
        if (currState != lastState || frame == m_to)
        {
            //draw lastState from lastStateFrame to frame.
            int startFrame = lastStateFrame - m_from;
            if (startFrame != 0)
                startFrame -= 1;
            int endFrame = frame - m_from;
            if (endFrame == startFrame)
                endFrame +=1;
            else if (frame != m_to)
                endFrame -= 1;
            int x1 = _frameToPos(startFrame);
            int x2 = _frameToPos(endFrame);
            int y = height() - scaleH;

            QRect rec(x1, y, x2 - x1 + 1, scaleH);
            painter.setPen(Qt::NoPen);

            if (lastState == zeno::GlobalComm::FRAME_COMPLETED)
            {
                painter.fillRect(rec, QColor(169, 169, 169, 100));
            }
            else if (lastState == zeno::GlobalComm::FRAME_BROKEN)
            {
                painter.fillRect(rec, QColor(255, 0, 0, 100));
            }
            lastState = currState;
            lastStateFrame = frame;
        }
    }
}

void ZSlider::drawSlideHandle(QPainter* painter, int scaleH)
{
    //draw time slider
    qreal xleftmost = _frameToPos(0);
    qreal xrightmost = _frameToPos(m_to - m_from);
    qreal xarrow_pos = _frameToPos(m_value - m_from);

    painter->setPen(Qt::NoPen);
    int y = height() - scaleH;
    //no need to show the progress of play frames.
    //painter->fillRect(QRectF(QPointF(xleftmost, y), QPointF(xarrow_pos, y + scaleH)), QColor(76, 159, 244, 64));

    //draw handle.
    static const int handleHeight = ZenoStyle::dpiScaled(12);
    static const int handleWidth = ZenoStyle::dpiScaled(6);
    y = height() - handleHeight;
    qreal x = xarrow_pos - handleWidth / 2;
    painter->fillRect(QRectF(QPointF(xarrow_pos - handleWidth / 2, y),
                             QPointF(xarrow_pos + handleWidth / 2, y + handleHeight)),
                      QColor(76, 159, 244));

    QFont font = QApplication::font();
    font.setPointSize(10);
    QFontMetrics metrics(font);
    painter->setFont(font);

    QString numText = QString::number(m_value);
    int w = metrics.horizontalAdvance(numText);
    painter->setPen(QColor(76, 159, 244));
    painter->drawText(QPointF(xarrow_pos - w / 2, height() * 0.4), numText);
}

int ZSlider::getCellLength(int total) {
    if (total < 20)
    {
        return 1;
    }
    int cellPixelLength = 0;
    int times = 1;
    int last = 0;
    int len;
    while (true)
    {
        for (int i = 0; i < 3; i++)
        {
            len = m_lengthUnit[i] * times;
            cellPixelLength = width() / ((total / len) + 1);
            if (150 < cellPixelLength && cellPixelLength <= 250)
            {
                return len;
            } else if (cellPixelLength > 250)
            {
                return last;
            }
            last = len;
        }
        if (INT_MAX / 10 < times)
        {
            return last;
        }
        times *= 10;
    }
}