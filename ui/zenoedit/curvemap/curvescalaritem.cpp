#include <QtWidgets>
#include "curvescalaritem.h"
#include "curvemapview.h"
#include "curveutil.h"


CurveScalarItem::CurveScalarItem(bool bHorizontal, CurveMapView* pView, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_bHorizontal(bHorizontal)
	, m_view(pView)
{
	setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

QRectF CurveScalarItem::boundingRect() const
{
	const QMargins margins = m_view->margins();
    const QTransform& trans = m_view->transform();
	const QRectF& rcGrid = m_view->gridBoundingRect();
	if (m_bHorizontal)
	{
		qreal height = sz;
		qreal width = rcGrid.width();
		QRectF orginalRc(rcGrid.left(), 0, width * trans.m11(), height);
		return orginalRc;
	}
	else
	{
		qreal width = sz;
		qreal height = rcGrid.height();
        QRectF orginalRc(0, rcGrid.top(), width, height * trans.m22());
        return orginalRc;
	}
}

void CurveScalarItem::resetPosition()
{
	QRect rcViewport = m_view->viewport()->rect();
	QPointF wtf = m_view->mapToScene(rcViewport.topLeft());
	const QTransform& trans = m_view->transform();
	const QMargins& margins = m_view->margins();
	if (m_bHorizontal)
	{
		wtf = m_view->mapToScene(rcViewport.bottomLeft() - QPoint(0, sz));
        setX(margins.left());
		setY(wtf.y());
	}
	else
	{
		setX(wtf.x());
        setY(margins.top());
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

	if (n <= 0)
		return;

	const QTransform &trans = m_view->transform();
    qreal scaleX = trans.m11();
    qreal scaleY = trans.m22();

	int prec = 2;
	int nFrames = 0;
	if (m_bHorizontal)
	{
        nFrames = m_view->frames(true);
	}
	else
	{
        nFrames = m_view->frames(false);
	}

    auto frames = curve_util::numframes(trans.m11(), trans.m22());
    nFrames = m_bHorizontal ? frames.first : frames.second;

	CURVE_RANGE rg = m_view->range();

	QFont font("HarmonyOS Sans", 12);
	QFontMetrics metrics(font);
	painter->setFont(font);
	painter->setPen(QPen(QColor(153, 153, 153)));

	qreal epsilon = 16;

	if (m_bHorizontal)
	{
		qreal realWidth = rcGrid.width() * scaleX;
        n *= scaleX;
		qreal step = realWidth / nFrames;
		qreal y = option->rect.top();
		for (qreal i = 0; i < realWidth; i += step)
		{
			qreal x = i;
			qreal scalar = (rg.xTo - rg.xFrom) * (i - 0) / n + rg.xFrom;
			
			QString numText = QString::number(scalar, 'g', prec);
			qreal textWidth = metrics.horizontalAdvance(numText);
			painter->drawText(QPoint(x - textWidth / 2, y + 22), numText);
		}

		qreal x = realWidth;
		qreal scalar = rg.xTo;
		QString numText = QString::number(scalar, 'g', prec);
		qreal textWidth = metrics.horizontalAdvance(numText);
		painter->drawText(QPoint(x - textWidth / 2, y + 22), numText);
	}
	else
	{
		qreal realHeight = rcGrid.height() * trans.m22();
		n *= scaleY;
		qreal step = realHeight / nFrames;
		qreal x = option->rect.left();
		for (qreal i = realHeight; i > 0; i -= step)
		{	
			qreal y = i;
			qreal scalar = (rg.yTo - rg.yFrom) * (n - i) / n + rg.yFrom;
			QString numText = QString::number(scalar, 'g', prec);
			qreal textHeight = metrics.height();

			painter->drawText(QPointF(x + 10, y + textHeight / 2), numText);
		}

		qreal y = 0;
		qreal scalar = rg.yTo;
		QString numText = QString::number(scalar, 'g', prec);
		qreal textHeight = metrics.height();
		painter->drawText(QPointF(x + 10, y + textHeight / 2), numText);
	}
}