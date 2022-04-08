#include "curvegrid.h"
#include "curvemapview.h"
#include "curvenodeitem.h"
#include "curveutil.h"
#include <zenoui/util/uihelper.h>
#include <QtGui/private/qbezier_p.h>
#include <QtGui/private/qstroker_p.h>

using namespace curve_util;

CurveGrid::CurveGrid(CurveMapView* pView, const QRectF& rc, QGraphicsItem* parent)
	: QGraphicsObject(parent)
	, m_view(pView)
	, m_initRc(rc)
	, m_bFCurve(true)
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
        connect(pNodeItem, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));

		if (i == 0)
		{
            m_vecNodes.append(pNodeItem);
			continue;
		}

		CurvePathItem *pathItem = new CurvePathItem(this);
        connect(pathItem, SIGNAL(clicked(const QPointF &)), this, SLOT(onPathClicked(const QPointF&)));

		QPainterPath path;

		QPointF lastNodePos = m_invTrans.map(pts[i-1]);
		QPointF lastRightPos = m_invTrans.map(pts[i-1] + handlers[(i - 1) * 2 + 1]);

		path.moveTo(lastNodePos);
		path.cubicTo(lastRightPos, leftScenePos, scenePos);
		pathItem->setPath(path);
        pathItem->update();

		m_vecNodes.append(pNodeItem);
        m_vecCurves.append(pathItem);
	}
}

int CurveGrid::nodeCount() const
{
    return m_vecNodes.size();
}

int CurveGrid::indexOf(CurveNodeItem *pItem) const
{
    return m_vecNodes.indexOf(pItem);
}

QPointF CurveGrid::nodePos(int i) const
{
    Q_ASSERT(i >= 0 && i < m_vecNodes.size());
    return m_vecNodes[i]->pos();
}

CurveNodeItem* CurveGrid::nodeItem(int i) const
{
    Q_ASSERT(i >= 0 && i < m_vecNodes.size());
    return m_vecNodes[i];
}

void CurveGrid::onNodeGeometryChanged()
{
    CurveNodeItem* pNode = qobject_cast<CurveNodeItem*>(sender());
    int i = m_vecNodes.indexOf(pNode);
    Q_ASSERT(i >= 0);

    QGraphicsPathItem* pLeftCurve = i > 0 ? m_vecCurves[i-1] : nullptr;
    if (pLeftCurve)
	{
        CurveNodeItem *pLeftNode = m_vecNodes[i - 1];
        QPainterPath path;
        path.moveTo(pLeftNode->pos());
        path.cubicTo(pLeftNode->rightHandlePos(), pNode->leftHandlePos(), pNode->pos());
        pLeftCurve->setPath(path);
        pLeftCurve->update();
	}

	QGraphicsPathItem *pRightCurve = (i < m_vecNodes.size() - 1) ? m_vecCurves[i] : nullptr;
    if (pRightCurve)
	{
        CurveNodeItem *pRightNode = m_vecNodes[i + 1];
        QPainterPath path;
        path.moveTo(pNode->pos());
        path.cubicTo(pNode->rightHandlePos(), pRightNode->leftHandlePos(), pRightNode->pos());
        pRightCurve->setPath(path);
        pRightCurve->update();
	}
}

void CurveGrid::onNodeDeleted()
{
    CurveNodeItem* pItem = qobject_cast<CurveNodeItem*>(sender());
    Q_ASSERT(pItem);
    int i = m_vecNodes.indexOf(pItem);
    if (i == 0 || i == m_vecNodes.size() - 1)
        return;

	CurveNodeItem* pLeftNode = m_vecNodes[i - 1];
    CurveNodeItem* pRightNode = m_vecNodes[i + 1];

	//curves[i-1] as a new curve from node i-1 to node i.
	CurvePathItem* pathItem = m_vecCurves[i - 1];

    m_vecCurves[i]->deleteLater();
	pItem->deleteLater();

	CurvePathItem* pDeleleCurve = m_vecCurves[i];
	m_vecCurves.remove(i);
    m_vecNodes.remove(i);

	QPainterPath path;
    path.moveTo(pLeftNode->pos());
	path.cubicTo(pLeftNode->rightHandlePos(), pRightNode->leftHandlePos(), pRightNode->pos());
    pathItem->setPath(path);
	pathItem->update();
}

void CurveGrid::onPathClicked(const QPointF& pos)
{
	CurvePathItem* pItem = qobject_cast<CurvePathItem*>(sender());
    Q_ASSERT(pItem);
    int i = m_vecCurves.indexOf(pItem);
    CurveNodeItem *pLeftNode = m_vecNodes[i];
    CurveNodeItem *pRightNode = m_vecNodes[i + 1];

	QPointF leftNodePos = pLeftNode->pos(), rightHdlPos = pLeftNode->rightHandlePos(),
			leftHdlPos = pRightNode->leftHandlePos(), rightNodePos = pRightNode->pos();

	/*
	QBezier bezier = QBezier::fromPoints(leftNodePos, rightHdlPos, leftHdlPos, rightNodePos);
	qreal t = (pos.x() - leftNodePos.x()) / (rightNodePos.x() - leftNodePos.x());
	QPointF k = bezier.derivedAt(t);
    QVector2D vec(k);
    vec.normalize();
	*/

	QPointF leftOffset(-50, 0);
    QPointF rightOffset(50, 0);

	//insert a new node.
    CurveNodeItem* pNewNode = new CurveNodeItem(m_view, pos, this);
	connect(pNewNode, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));
    connect(pNewNode, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));

    pNewNode->initHandles(leftOffset, rightOffset);

	CurvePathItem* pLeftHalf = pItem;
    CurvePathItem* pRightHalf = new CurvePathItem(this);
	connect(pRightHalf, SIGNAL(clicked(const QPointF &)), this, SLOT(onPathClicked(const QPointF&)));

	QPainterPath leftPath;
    leftPath.moveTo(leftNodePos);
	leftPath.cubicTo(rightHdlPos, pNewNode->leftHandlePos(), pNewNode->pos());
    pLeftHalf->setPath(leftPath);
	pLeftHalf->update();

	QPainterPath rightPath;
    rightPath.moveTo(pNewNode->pos());
	rightPath.cubicTo(pNewNode->rightHandlePos(), leftHdlPos, rightNodePos);
    pRightHalf->setPath(rightPath);
	pRightHalf->update();

	m_vecNodes.insert(i + 1, pNewNode);
    m_vecCurves.insert(i + 1, pRightHalf);
}

bool CurveGrid::isFuncCurve() const
{
    return m_bFCurve;
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