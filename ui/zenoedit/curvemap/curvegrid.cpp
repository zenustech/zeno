#include "curvegrid.h"
#include "curvemapview.h"


CurveGrid::CurveGrid(CurveMapView* pView, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_view(pView)
	, m_sample(nullptr)
{
	//m_sample = new QGraphicsRectItem(250, 250, 100, 100, this);
	//m_sample->setBrush(QColor(255, 0, 0));

	//m_initRc = pView->rect().adjusted(20, 20, -20, -20);
}

void CurveGrid::setColor(const QColor& clrGrid, const QColor& clrBackground)
{
	m_clrGrid = clrGrid;
	m_clrBg = clrBackground;
}

void CurveGrid::updateTransform()
{
	QTransform trans;
	QRectF br = boundingRect();
}

QRectF CurveGrid::boundingRect() const
{
	//return m_view->gridBoundingRect();
	return m_initRc;

	CURVE_RANGE rg = m_view->range();
	return QRectF(QPointF(rg.xFrom, rg.yFrom), QPointF(rg.xTo, rg.yTo));

}

void CurveGrid::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QGraphicsObject::mousePressEvent(event);
	QPointF pos = event->pos();
	QPointF wtf = mapFromScene(pos);
	QPointF pos2 = m_transform.map(pos);
	int j;
	j = 0;
	if (event->button() == Qt::RightButton)
	{
		m_sample = new QGraphicsRectItem(0, 0, 64, 64, this);
		QPointF phyPos = m_transform.inverted().map(QPointF(0.5, 0.5));
		m_sample->setPos(phyPos);
		m_sample->setBrush(QColor(255, 0, 0));
	}
}

void CurveGrid::initRect(const QRectF& rc)
{
	if (!m_initRc.isValid())
	{
		m_initRc = rc;
		m_initRc.adjust(64, 64, -64, -64);

		//setup the transform.
		CURVE_RANGE rg = m_view->range();

		QPolygonF polygonIn;
		polygonIn << m_initRc.topLeft()
			<< m_initRc.topRight()
			<< m_initRc.bottomRight()
			<< m_initRc.bottomLeft();

		QPolygonF polygonOut;
		polygonOut << QPointF(rg.xFrom, rg.yFrom)
			<< QPointF(rg.xTo, rg.yFrom)
			<< QPointF(rg.xTo, rg.yTo)
			<< QPointF(rg.xFrom, rg.yTo);

		auto isOk = QTransform::quadToQuad(polygonIn, polygonOut, m_transform);
		if (!isOk)
		{
			return;
		}

		//QPointF pos1 = m_transform.map(m_initRc.topLeft());
		//QPointF pos2 = m_transform.map(m_initRc.bottomRight());
		//QPointF pos3 = m_transform.map(m_initRc.center());
		//setTransform(m_transform);
		//this->setTransformOriginPoint();
	}
}

void CurveGrid::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	//painter->fillRect(boundingRect(), QColor(255, 255, 255));
	//return;


	QVarLengthArray<QLineF, 256> innerLines;

	const QRectF& rc = boundingRect();

	int W = rc.width(), H = rc.height();
	qreal factor = m_view->factor();
	int nVLines = m_view->frames(true);
	int nHLines = m_view->frames(false);

	const qreal left = rc.left(), right = rc.right();
	const qreal top = rc.top(), bottom = rc.bottom();

	for (qreal x = left; x < right;)
	{
		innerLines.append(QLineF(x, top, x, bottom));
		qreal steps = qMax(1., (right - left) / nVLines);
		x += steps;
	}

	for (qreal y = top; y < bottom;)
	{
		innerLines.append(QLineF(left, y, right, y));
		qreal steps = qMax(1., (bottom - top) / nHLines);
		y += steps;
	}

	painter->fillRect(rc, m_clrBg);
	
	QPen pen(QColor(m_clrGrid), 1. / factor);
	painter->setPen(pen);

	painter->drawRect(rc);
	painter->drawLines(innerLines.data(), innerLines.size());
}