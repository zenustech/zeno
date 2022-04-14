#include "curvegrid.h"
#include "curvenodeitem.h"
#include "curvemapview.h"
#include "curvesitem.h"


CurvesItem::CurvesItem(CurveMapView* pView, CurveGrid* grid, const QRectF& rc, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_view(pView)
    , m_grid(grid)
{
}

CurvesItem::~CurvesItem()
{

}

void CurvesItem::initCurves(const QVector<QPointF>& pts, const QVector<QPointF>& handlers)
{
    int N = pts.size();
	Q_ASSERT(N * 2 == handlers.size());

	for (int i = 0; i < N; i++)
	{
		QPointF scenePos = m_grid->logicToScene(pts[i]);
		QPointF leftScenePos = m_grid->logicToScene(pts[i] + handlers[i * 2]);
		QPointF rightScenePos = m_grid->logicToScene(pts[i] + handlers[i * 2 + 1]);
		QPointF leftOffset = leftScenePos - scenePos;
		QPointF rightOffset = rightScenePos - scenePos;

		CurveNodeItem* pNodeItem = new CurveNodeItem(m_view, scenePos, m_grid, this);
		pNodeItem->initHandles(leftOffset, rightOffset);
		connect(pNodeItem, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));
        connect(pNodeItem, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));

		if (i == 0)
		{
            m_vecNodes.append(pNodeItem);
			continue;
		}

		CurvePathItem* pathItem = new CurvePathItem(this);
        connect(pathItem, SIGNAL(clicked(const QPointF&)), this, SLOT(onPathClicked(const QPointF&)));

		QPainterPath path;

		QPointF lastNodePos = m_grid->logicToScene(pts[i - 1]);
		QPointF lastRightPos = m_grid->logicToScene(pts[i-1] + handlers[(i - 1) * 2 + 1]);

		path.moveTo(lastNodePos);
		path.cubicTo(lastRightPos, leftScenePos, scenePos);
		pathItem->setPath(path);
        pathItem->update();

		m_vecNodes.append(pNodeItem);
        m_vecCurves.append(pathItem);
	}
}

int CurvesItem::indexOf(CurveNodeItem *pItem) const
{
    return m_vecNodes.indexOf(pItem);
}

int CurvesItem::nodeCount() const
{
    return m_vecNodes.size();
}

QPointF CurvesItem::nodePos(int i) const
{
    Q_ASSERT(i >= 0 && i < m_vecNodes.size());
    return m_vecNodes[i]->pos();
}

CurveNodeItem* CurvesItem::nodeItem(int i) const
{
    Q_ASSERT(i >= 0 && i < m_vecNodes.size());
    return m_vecNodes[i];
}

QRectF CurvesItem::boundingRect() const
{
    return childrenBoundingRect();
}

void CurvesItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
}

void CurvesItem::onNodeGeometryChanged()
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

	QGraphicsPathItem* pRightCurve = (i < m_vecNodes.size() - 1) ? m_vecCurves[i] : nullptr;
    if (pRightCurve)
	{
        CurveNodeItem* pRightNode = m_vecNodes[i + 1];
        QPainterPath path;
        path.moveTo(pNode->pos());
        path.cubicTo(pNode->rightHandlePos(), pRightNode->leftHandlePos(), pRightNode->pos());
        pRightCurve->setPath(path);
        pRightCurve->update();
	}

    emit nodesDataChanged();
}

void CurvesItem::onNodeDeleted()
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

	emit nodesDataChanged();
}

void CurvesItem::onPathClicked(const QPointF& pos)
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
    CurveNodeItem* pNewNode = new CurveNodeItem(m_view, pos, m_grid, this);
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