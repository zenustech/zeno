#include <QtWidgets>
#include "curvescalaritem.h"
#include "curvemapview.h"


CurveScalarItem::CurveScalarItem(bool bHorizontal, ZCurveMapView* pView, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_bHorizontal(bHorizontal)
	, m_view(pView)
{
	setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

QRectF CurveScalarItem::boundingRect() const
{
	const QMargins margins = m_view->margins();
	const QRectF& rcGrid = m_view->gridBoundingRect();
	if (m_bHorizontal)
	{
		qreal height = sz;
		qreal width = rcGrid.width();
		return QRectF(rcGrid.left(), 0, width * m_view->factor(), height);
	}
	else
	{
		qreal width = sz;
		qreal height = rcGrid.height();
		return QRectF(0, rcGrid.top(), width, height * m_view->factor());
	}
}

void CurveScalarItem::resetPosition()
{
	QRect rcViewport = m_view->viewport()->rect();
	QPointF wtf = m_view->mapToScene(rcViewport.topLeft());
	if (m_bHorizontal)
	{
		wtf = m_view->mapToScene(rcViewport.bottomLeft());
		const QMargins& margins = m_view->margins();
		setY(wtf.y() - sz / m_view->factor());
	}
	else
	{
		setX(wtf.x());
	}
	m_view->scene()->update();
}

void CurveScalarItem::update()
{
	resetPosition();
	_base::update();
}

void CurveScalarItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	qreal n = 0;

	int from = 0, to = 0;
	QRectF rc = m_view->rect();
	const QRectF& rcGrid = m_view->gridBoundingRect();
	if (m_bHorizontal)
	{
		n = rcGrid.width();
		from = 0;
		to = n;
	}
	else
	{
		n = rcGrid.height();
		from = 0;
		to = n;
	}

	qreal factor = m_view->factor();

	if (n <= 0)
		return;

	int nFrames = 0;
	if (m_bHorizontal)
		nFrames = m_view->frames(true);
	else
		nFrames = m_view->frames(false);

	CURVE_RANGE rg = m_view->range();

	QFont font("HarmonyOS Sans", 12);
	QFontMetrics metrics(font);
	painter->setFont(font);
	painter->setPen(QPen(QColor(153, 153, 153)));

	qreal epsilon = 16;

	if (m_bHorizontal)
	{
		qreal step = rcGrid.width() / nFrames;
		qreal y = option->rect.top();
		for (qreal i = 0; i < rcGrid.width(); i += step)
		{
			qreal x = i * factor;
			qreal scalar = (rg.xTo - rg.xFrom) * (i - 0) / n + rg.xFrom;
			int prec = 3;
			QString numText = QString::number(scalar, 'g', prec);
			qreal textWidth = metrics.horizontalAdvance(numText);
			painter->drawText(QPoint(x - textWidth / 2, y + 22), numText);
		}

		qreal x = rcGrid.width() * factor;
		qreal scalar = rg.xTo;
		int prec = 3;
		QString numText = QString::number(scalar, 'g', prec);
		qreal textWidth = metrics.horizontalAdvance(numText);
		painter->drawText(QPoint(x - textWidth / 2, y + 22), numText);
	}
	else
	{
		qreal step = rcGrid.height() / nFrames;
		qreal x = option->rect.left();
		for (qreal i = rcGrid.height(); i > 0; i -= step)
		{	
			qreal y = i * factor;
			qreal scalar = (rg.yTo - rg.yFrom) * (n - i) / n;
			int prec = 3;
			QString numText = QString::number(scalar, 'g', prec);
			qreal textHeight = metrics.height();

			painter->drawText(QPointF(x + 10, y + textHeight / 2), numText);
		}

		qreal y = 0;
		qreal scalar = rg.yTo;
		int prec = 3;
		QString numText = QString::number(scalar, 'g', prec);
		qreal textHeight = metrics.height();
		painter->drawText(QPointF(x + 10, y + textHeight / 2), numText);
	}
}