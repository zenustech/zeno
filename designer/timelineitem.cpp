#include "framework.h"
#include "timelineitem.h"
#include "nodescene.h"


TimelineItemTemp::TimelineItemTemp(QRectF rcView, QGraphicsItem* parent)
	: QGraphicsRectItem(0, 0, rcView.width(), 20, parent)
{
	setPen(Qt::NoPen);
	setBrush(QColor(51, 51, 51));
}

void TimelineItemTemp::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	QGraphicsRectItem::paint(painter, option, widget);
}


///////////////////////////////////////////////////////////////
TimelineItem::TimelineItem(NodeScene* pScene, bool bHorizontal, QRectF rcView, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_view(rcView)
    , m_bHorizontal(bHorizontal)
    , m_pScene(pScene)
    , m_left(0)
    , m_top(0)
    , m_right(0)
    , m_bottom(0)
    , m_from(0)
    , m_to(0)
    , m_nframes(3000)
    , m_factor(1.)
{
    m_left = m_view.left();
    m_right = m_view.right();
    m_top = m_view.top();
    m_bottom = m_view.bottom();
    if (m_bHorizontal)
    {
        m_from = m_left; m_to = m_right;
        m_nframes = m_view.width() / 100 * 10;
    }
    else
    {
        m_from = m_top; m_to = m_bottom;
        m_nframes = m_view.height() / 100 * 10;
    }
    updateScalar(1.0);

    setZValue(100);

    QList<QGraphicsView*> pViews = m_pScene->views();
    QGraphicsView* pView = pViews[0];
    connect(pView->horizontalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(resetPosition()));
    connect(pView->verticalScrollBar(), SIGNAL(valueChanged(int)), this, SLOT(resetPosition()));
    setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

void TimelineItem::resetPosition()
{
    QList<QGraphicsView*> pViews = m_pScene->views();
    QGraphicsView* pView = pViews[0];
    QRect rcViewport = pView->viewport()->rect();
    QPointF wtf = pView->mapToScene(rcViewport.topLeft());
    if (m_bHorizontal)
        setY(wtf.y());
    else
        setX(wtf.x());
    m_pScene->update();
}

void TimelineItem::updateScalar(qreal factor)
{
    m_factor = factor;
    int nLength = m_bHorizontal ? m_view.width() : m_view.height();
    if (factor < 0.5)
    {
        m_nframes = nLength / 100;
    }
    else if (factor < 1.)
    {
        m_nframes = nLength / 100 * 5 * 0.5;
    }
    else if (factor == 1.)
    {
        m_nframes = nLength / 100 * 10 * 0.5;
    }
    else if (factor < 2)
    {
        m_nframes = nLength / 100 * 10 * 2 * 0.5;
    }
    else if (factor < 3)
    {
        m_nframes = nLength / 100 * 10 * 4 * 0.5;
    }
    else if (factor < 4)
    {
        m_nframes = nLength / 100 * 10 * 10 * 0.5;
    }
    else
    {
        m_nframes = nLength / 100 * 10 * 15 * 0.5;
    }
    resetPosition();
    return;
}

QRectF TimelineItem::boundingRect() const
{
    if (m_bHorizontal)
        return QRectF(m_left, 0, m_view.width(), sz);
    else
        return QRectF(0, m_top, sz, m_view.height());
}

void TimelineItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//fill background
	painter->fillRect(option->rect, QColor(51, 51, 51));

    int n = m_to - m_from + 1;

    QFont font("Calibre", 9);
    QFontMetrics metrics(font);
    painter->setFont(font);
    painter->setPen(QPen(QColor(153, 153, 153)));

    for (int i = m_from, j = 0; i <= m_to; i += (n / m_nframes), j++)
    {
        int h = 0;
        bool midScalar = j % 10 == 0;
        if (midScalar)
        {
            h = 13;
        }
        else
        {
            h = 5;
        }

        if (m_bHorizontal)
        {
            qreal x = i * m_factor;
            qreal y = option->rect.top();
            painter->drawLine(QPointF(x, y), QPointF(x, y + h));
            if (midScalar)
            {
                painter->drawText(QPoint(x + 7, y + 22), QString::number(i));
            }
        }
        else
        {
            qreal x = option->rect.left();
            qreal y = i * m_factor;
            painter->drawLine(QPointF(x, y), QPointF(x + h, y));
            if (midScalar)
            {
                painter->save();
                painter->translate(24, y + 12);
                painter->rotate(-90);
                painter->drawText(QPointF(x + h + 5, 0), QString::number(i));
                painter->restore();
            }
        }
    }
}