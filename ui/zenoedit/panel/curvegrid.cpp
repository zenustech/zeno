#include "curvegrid.h"


CurveGrid::CurveGrid(QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_factor(1.)
	, m_nFrames(20)
{

}

void CurveGrid::reset(const QRectF& rc)
{
	m_sceneRect = rc;
	update();
}

void CurveGrid::setColor(const QColor& clrGrid, const QColor& clrBackground)
{
	m_clrGrid = clrGrid;
	m_clrBg = clrBackground;
}

void CurveGrid::setFactor(const qreal& factor, int nFrames)
{
	m_factor = factor;
	m_nFrames = nFrames;
	update();
}

QRectF CurveGrid::boundingRect() const
{
	return m_sceneRect;
}

void CurveGrid::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	QVarLengthArray<QLineF, 256> innerLines;
	int nHLines = 10;
	int nVLines = 10;
	int W = m_sceneRect.width(), H = m_sceneRect.height();

	int nLines = m_nFrames;

	qreal left = m_sceneRect.left(), right = m_sceneRect.right();
	for (qreal x = left; x <= right;)
	{
		innerLines.append(QLineF(x, m_sceneRect.top(), x, m_sceneRect.bottom()));
		qreal steps = qMax(1., (right - left) / nLines);
		x += steps;
	}

	qreal top = m_sceneRect.top(), bottom = m_sceneRect.bottom();
	for (qreal y = top; y <= bottom;)
	{
		innerLines.append(QLineF(m_sceneRect.left(), y, m_sceneRect.right(), y));
		qreal steps = qMax(1., (bottom - top) / nLines);
		y += steps;
	}

	painter->fillRect(m_sceneRect, m_clrBg);
	
	QPen pen(QColor(m_clrGrid), 1./m_factor);
	painter->setPen(pen);
	painter->drawLines(innerLines.data(), innerLines.size());
}