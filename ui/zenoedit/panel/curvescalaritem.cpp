#include <QtWidgets>
#include "curvescalaritem.h"


CurveScalarItem::CurveScalarItem(bool bHorizontal, CURVE_RANGE rg, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_bHorizontal(bHorizontal)
	, m_from(0)
	, m_to(0)
	, m_nframes(30)
	, m_factor(1.)
	, m_range(rg)
{
	if (m_bHorizontal)
	{
		m_from = rg.xFrom;
		m_to = rg.xTo;
	}
	else
	{
		m_from = rg.yFrom;
		m_to = rg.yTo;
	}
	setFlag(QGraphicsItem::ItemIgnoresTransformations);
}

//QRectF CurveScalarItem::boundingRect() const
//{
//	static const int margin = 64;
//	if (m_bHorizontal)
//	{
//		qreal height = sz / m_factor;
//		return QRectF(margin, 0, m_view.width() - 2 * margin, height);
//	}
//	else
//	{
//		qreal width = sz / m_factor;
//		return QRectF(0, margin, width, m_view.height() - 2 * margin);
//	}
//}

QRectF CurveScalarItem::boundingRect() const
{
	static const int margin = 64;
	if (m_bHorizontal)
	{
		qreal height = sz;
		return QRectF(margin, 0, m_view.width() * m_factor, height);
	}
	else
	{
		qreal width = sz;
		return QRectF(0, margin, width, m_view.height() * m_factor);
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
	m_view = pView->rect();
	if (m_bHorizontal)
	{
		m_from = m_view.left();
		m_to = m_view.right();
	}
	else
	{
		m_from = m_view.top();
		m_to = m_view.bottom();
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
	qreal n = m_to - m_from - 2 * 64;

	QFont font("Calibre", 9);
	QFontMetrics metrics(font);
	painter->setFont(font);
	painter->setPen(QPen(QColor(153, 153, 153)));

	for (qreal i = m_from, j = 0; i <= m_to; i += (n / m_nframes), j++)
	{
		int h = 0;
		bool midScalar = false;// j % 10 == 0;
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