#include "curvegrid.h"
#include "curvemapview.h"
#include "curvenodeitem.h"
#include "curveutil.h"
#include <zenoui/util/uihelper.h>

using namespace curve_util;

CurveGrid::CurveGrid(CurveMapView* pView, const QRectF& rc, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_view(pView)
	, m_initRc(rc)
{
	initTransform();
}

void CurveGrid::initTransform()
{
	m_initRc.adjust(64, 64, -64, -64);

	//setup the transform.
	CURVE_RANGE rg = m_view->range();

	QPolygonF polygonIn;
	polygonIn << m_initRc.topLeft()
		<< m_initRc.topRight()
		<< m_initRc.bottomRight()
		<< m_initRc.bottomLeft();

	QPolygonF polygonOut;
	polygonOut << QPointF(rg.xFrom, rg.yTo)
		<< QPointF(rg.xTo, rg.yTo)
		<< QPointF(rg.xTo, rg.yFrom)
		<< QPointF(rg.xFrom, rg.yFrom);

	bool isOk = QTransform::quadToQuad(polygonIn, polygonOut, m_transform);
	if (!isOk)
	{
		Q_ASSERT(false);
		return;
	}

	m_invTrans = m_transform.inverted(&isOk);
	if (!isOk)
	{
		Q_ASSERT(false);
		return;
	}

	QPointF pos1 = m_transform.map(m_initRc.topLeft());
	QPointF pos2 = m_transform.map(m_initRc.bottomRight());
	QPointF pos3 = m_transform.map(m_initRc.center());
}

void CurveGrid::initCurves(const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
	int N = pts.size();
	Q_ASSERT(N * 2 == handlers.size());

	for (int i = 0; i < N; i++)
	{
		QPointF scenePos = m_invTrans.map(pts[i]);
		QPointF leftScenePos = m_invTrans.map(pts[i] + handlers[i * 2]);
		QPointF rightScenePos = m_invTrans.map(pts[i] + handlers[i * 2 + 1]);
		QPointF leftOffset = leftScenePos - scenePos;
		QPointF rightOffset = rightScenePos - scenePos;

		CurveNodeItem* pNodeItem = new CurveNodeItem(m_view, scenePos, this);
		pNodeItem->initHandles(leftOffset, rightOffset);
		connect(pNodeItem, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));

		if (i == 0)
		{
            m_vecNodes.append(pNodeItem);
			continue;
		}

		QGraphicsPathItem* pathItem = new QGraphicsPathItem(this);
		const int penWidth = 2;
		QPen pen(QColor(231, 29, 31), penWidth);
		pen.setStyle(Qt::SolidLine);
		pathItem->setPen(pen);

		QPainterPath path;

		QPointF lastNodePos = m_invTrans.map(pts[i-1]);
		QPointF lastRightPos = m_invTrans.map(pts[i-1] + handlers[(i - 1) * 2 + 1]);

		path.moveTo(lastNodePos);
		path.cubicTo(lastRightPos, leftScenePos, scenePos);
		pathItem->setPath(path);
		pathItem->update();

		pNodeItem->setLeftCurve(pathItem);
        m_vecNodes[i-1]->setRightCurve(pathItem);

		m_vecNodes.append(pNodeItem);
	}
}

void CurveGrid::onNodeGeometryChanged()
{
    CurveNodeItem* pNode = qobject_cast<CurveNodeItem*>(sender());
    int i = m_vecNodes.indexOf(pNode);
    Q_ASSERT(i >= 0);

    QGraphicsPathItem* pLeftCurve = pNode->leftCurve();
    if (pLeftCurve)
	{
        Q_ASSERT(i >= 1);
        CurveNodeItem *pLeftNode = m_vecNodes[i - 1];

        QPainterPath path;
        path.moveTo(pLeftNode->pos());
        path.cubicTo(pLeftNode->rightHandlePos(), pNode->leftHandlePos(), pNode->pos());
        pLeftCurve->setPath(path);
        pLeftCurve->update();
	}

	QGraphicsPathItem* pRightCurve = pNode->rightCurve();
    if (pRightCurve)
	{
        Q_ASSERT(i < m_vecNodes.size() - 1);
        CurveNodeItem *pRightNode = m_vecNodes[i + 1];

        QPainterPath path;
        path.moveTo(pNode->pos());
        path.cubicTo(pNode->rightHandlePos(), pRightNode->leftHandlePos(), pRightNode->pos());
        pRightCurve->setPath(path);
        pRightCurve->update();
	}
}

void CurveGrid::setColor(const QColor& clrGrid, const QColor& clrBackground)
{
	m_clrGrid = clrGrid;
	m_clrBg = clrBackground;
}

QRectF CurveGrid::boundingRect() const
{
	return m_initRc;
}

void CurveGrid::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QGraphicsObject::mousePressEvent(event);
	QPointF pos = event->pos();
	QPointF wtf = mapFromScene(pos);
	QPointF pos2 = m_transform.map(pos);
}

void CurveGrid::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
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