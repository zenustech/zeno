#include <QtWidgets>
#include "curvescalaritem.h"
#include "zcurvemapeditor.h"


CurveScalarItem::CurveScalarItem(bool bHorizontal, ZCurveMapView* pView, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_bHorizontal(bHorizontal)
	, m_from(0)
	, m_to(0)
	, m_nframes(30)
	, m_factor(1.)
	, m_view(pView)
{
	if (m_bHorizontal)
	{
		m_from = pView->range().xFrom;
		m_to = pView->range().xTo;
	}
	else
	{
		m_from = pView->range().yFrom;
		m_to = pView->range().yTo;
	}
	setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

QRectF CurveScalarItem::boundingRect() const
{
	static const int margin = 64;
	if (m_bHorizontal)
	{
		qreal height = sz;
		qreal width = m_rect.width() - 2 * margin;
		return QRectF(margin, 0, width * m_factor, height);
	}
	else
	{
		qreal width = sz;
		return QRectF(0, margin, width, m_rect.height() * m_factor);
	}
}

void CurveScalarItem::resetPosition(QGraphicsView* pView)
{
	QRect rcViewport = pView->viewport()->rect();
	QPointF wtf = pView->mapToScene(rcViewport.topLeft());
	if (m_bHorizontal)
	{
		setY(wtf.y());
	}
	else
	{
		setX(wtf.x());
	}
	pView->scene()->update();
}

void CurveScalarItem::onResizeView(QGraphicsView* pView)
{
	m_rect = pView->rect();
	if (m_bHorizontal)
	{
		m_from = m_rect.left();
		m_to = m_rect.right();
	}
	else
	{
		m_from = m_rect.top();
		m_to = m_rect.bottom();
	}
	update();
}

void CurveScalarItem::updateScalar(QGraphicsView* pView, qreal factor, int nFrames)
{
	m_factor = factor;
	m_nframes = nFrames;
	resetPosition(pView);
}

void CurveScalarItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//fill background
	//painter->fillRect(option->rect, QColor(51, 51, 51));

	//return;
	static const int margin = 64;
	qreal n = 0;
	if (m_bHorizontal)
	{
		n = m_rect.width() - 2 * margin;
		m_from = 0;
		m_to = n;
	}
	else
	{
		n = m_rect.height() - 2 * margin;
		m_from = 0;
		m_to = n;
	}

	CURVE_RANGE rg = m_view->range();

	QFont font("HarmonyOS Sans", 12);
	QFontMetrics metrics(font);
	painter->setFont(font);
	painter->setPen(QPen(QColor(153, 153, 153)));

	for (qreal i = m_from, j = 0; i <= m_to; i += (n / m_nframes), j++)
	{
		if (m_bHorizontal)
		{
			qreal x = i * m_factor;
			qreal y = option->rect.top();

			qreal scalar = (rg.xTo - rg.xFrom) * (i - m_from) / n + rg.xFrom;

			QString numText = QString::number(scalar, 'g', 3);
			qreal textWidth = metrics.horizontalAdvance(numText);

			painter->drawText(QPoint(x - textWidth / 2, y + 22), numText);
		}
		else
		{
			qreal x = option->rect.left();
			qreal y = i * m_factor;

			qreal scalar = (rg.yTo - rg.yFrom) * (i - m_from) / n + rg.yFrom;
			QString numText = QString::number(scalar, 'g', 3);
			qreal textHeight = metrics.height();

			painter->drawText(QPointF(x + 10, y + textHeight / 2), numText);
		}
	}
}