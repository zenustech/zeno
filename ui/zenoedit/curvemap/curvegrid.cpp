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
	, m_model(nullptr)
	, m_selection(nullptr)
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

	m_model = new QStandardItemModel(this);
	m_selection = new QItemSelectionModel(m_model);
	MODEL_PACK pack;
	pack.pModel = m_model;
	pack.pSelection = m_selection;

	for (int i = 0; i < N; i++)
	{
		QPointF scenePos = m_invTrans.map(pts[i]);
		QPointF leftScenePos = m_invTrans.map(pts[i] + handlers[i * 2]);
		QPointF rightScenePos = m_invTrans.map(pts[i] + handlers[i * 2 + 1]);
		QPointF leftOffset = leftScenePos - scenePos;
		QPointF rightOffset = rightScenePos - scenePos;

		QStandardItem* pNode = new QStandardItem;
		const QString& id = UiHelper::generateUuid();
		pNode->setData(id, ROLE_ItemObjId);
		pNode->setData(ITEM_NODE, ROLE_ItemType);
		pNode->setData(scenePos, ROLE_ItemPos);
		pNode->setData(ITEM_UNTOGGLED, ROLE_ItemStatus);
		m_model->appendRow(pNode);

		const QModelIndex& idx = m_model->indexFromItem(pNode);
		CurveNodeItem* pNodeItem = new CurveNodeItem(m_view, scenePos, this);
		pNodeItem->initHandles(pack, idx, leftOffset, rightOffset);

		m_nodes.insert(id, pNodeItem);
	}

	//init curves.

	connect(pack.pModel, &QStandardItemModel::dataChanged, this, &CurveGrid::onDataChanged);
}

void CurveGrid::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
	ItemType type =(ItemType)topLeft.data(ROLE_ItemType).toInt();
	if (type == ITEM_NODE)
	{
		const QString& objId = topLeft.data(ROLE_ItemObjId).toString();
		if (m_nodes.find(objId) != m_nodes.end())
		{
			CurveNodeItem* pNodeItem = m_nodes[objId];
			pNodeItem->updateStatus();
		}
	}
	else if (type == ITEM_LEFTHANDLE || type == ITEM_RIGHTHANDLE)
	{
		const QString& objId = topLeft.data(ROLE_ItemObjId).toString();
		const QString& nodeId = topLeft.data(ROLE_ItemBelongTo).toString();
		auto lst = m_model->match(m_model->index(0, 0), ROLE_ItemObjId, nodeId, 1, Qt::MatchExactly);
		Q_ASSERT(lst.size() == 1);
		const QModelIndex& nodeIdx = lst[0];

		if (m_nodes.find(nodeId) != m_nodes.end())
		{
			CurveNodeItem* pNodeItem = m_nodes[nodeId];
			pNodeItem->updateHandleStatus(objId);
		}
	}
	else if (type == ITEM_CURVE)
	{

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