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
{
    m_left = m_view.left();
    m_right = m_view.right();
    m_top = m_view.top();
    m_bottom = m_view.bottom();
    if (m_bHorizontal)
    {
        m_from = m_left; m_to = m_right;
    }
    else
    {
        m_from = m_top; m_to = m_bottom;
    }

    setZValue(100);
    //connect(m_pScene, SIGNAL(changed(QList<QRectF>)), this, SLOT(resetPosition())); //refresh too slow.

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

QRectF TimelineItem::boundingRect() const
{
    if (m_bHorizontal)
        return QRectF(m_left, 0, m_view.width(), sz);
    else
        return QRectF(0, m_top, sz, m_view.height());
}

int TimelineItem::_getframes()
{
    int frames = 0;
    if (m_bHorizontal)
        frames = m_view.width() / 100 * 10;
    else
        frames = m_view.height() / 100 * 10;
    frames = m_view.width() / 100 * 10;
    return frames;
}

void TimelineItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//fill background
	painter->fillRect(option->rect, QColor(51, 51, 51));

    int n = m_to - m_from + 1;
    int frames = _getframes();

    QFont font("Calibre", 9);
    QFontMetrics metrics(font);
    painter->setFont(font);
    painter->setPen(QPen(QColor(153, 153, 153)));

    for (int i = m_from, j = 0; i <= m_to; i += (n / frames), j++)
    {
        int h = 0;
        if (j % 10 == 0)
        {
            h = 13;
        }
        else
        {
            h = 5;
        }

        if (m_bHorizontal)
        {
            int y = option->rect.top();
            painter->drawLine(QPointF(i, y), QPointF(i, y + h));
        }
        else
        {
            int x = option->rect.left();
            painter->drawLine(QPointF(x, i), QPointF(x + h, i));
        }
    }
}