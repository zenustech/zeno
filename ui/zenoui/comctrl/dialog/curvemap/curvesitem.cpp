#include "curvegrid.h"
#include "curvenodeitem.h"
#include "curvemapview.h"
#include "curvesitem.h"
#include <zenomodel/include/curvemodel.h>
#include <zenomodel/include/uihelper.h>
#include "zassert.h"


CurvesItem::CurvesItem(CurveMapView* pView, CurveGrid* grid, const QRectF& rc, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_view(pView)
    , m_grid(grid)
    , m_model(nullptr)
{
    m_color = QColor(77, 77, 77);
}

CurvesItem::~CurvesItem()
{
    //todo: delete all nodes and curves.
    for (auto i : m_vecNodes)
    {
        delete i;
    }
}

void CurvesItem::initCurves(CurveModel* model)
{
    m_model = model;

    for (int r = 0; r < m_model->rowCount(); r++)
    {
        QModelIndex idx = m_model->index(r, 0);
        QPointF logicPos = m_model->data(idx, ROLE_NODEPOS).toPointF();
        QPointF left = m_model->data(idx, ROLE_LEFTPOS).toPointF();
        QPointF right = m_model->data(idx, ROLE_RIGHTPOS).toPointF();

        QPointF scenePos = m_grid->logicToScene(logicPos);
        QPointF leftScenePos = m_grid->logicToScene(logicPos + left);
        QPointF rightScenePos = m_grid->logicToScene(logicPos + right);
		QPointF leftOffset = leftScenePos - scenePos;
		QPointF rightOffset = rightScenePos - scenePos;

		CurveNodeItem* pNodeItem = new CurveNodeItem(idx, m_view, scenePos, m_grid, this);
		pNodeItem->initHandles(leftOffset, rightOffset);
		connect(pNodeItem, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));
        connect(pNodeItem, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));

		if (r == 0)
		{
            m_vecNodes.append(pNodeItem);
			continue;
		}

		CurvePathItem* pathItem = new CurvePathItem(m_color, this);
        connect(pathItem, SIGNAL(clicked(const QPointF&)), this, SLOT(onPathClicked(const QPointF&)));

		QPainterPath path;

        idx = m_model->index(r - 1, 0);
        logicPos = m_model->data(idx, ROLE_NODEPOS).toPointF();
        right = m_model->data(idx, ROLE_RIGHTPOS).toPointF();
		QPointF lastNodePos = m_grid->logicToScene(logicPos);
        QPointF lastRightPos = m_grid->logicToScene(logicPos + right);

		path.moveTo(lastNodePos);
		path.cubicTo(lastRightPos, leftScenePos, scenePos);
		pathItem->setPath(path);
        pathItem->update();

		m_vecNodes.append(pNodeItem);
        m_vecCurves.append(pathItem);
    }

    connect(m_model, &CurveModel::dataChanged, this, &CurvesItem::onDataChanged);
    connect(m_model, &CurveModel::rowsInserted, this, &CurvesItem::onNodesInserted);
    connect(m_model, &CurveModel::rowsAboutToBeRemoved, this, &CurvesItem::onNodesAboutToBeRemoved);
}

void CurvesItem::onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles)
{
    //model sync to view(items).

    int r = topLeft.row();
    ZASSERT_EXIT(r >= 0 && r < m_vecNodes.size());
    CurveNodeItem *pNode = m_vecNodes[r];

    ZASSERT_EXIT(!roles.isEmpty());
    int role = roles[0];
    QPointF logicNodePos = topLeft.data(ROLE_NODEPOS).toPointF();
    QPointF sceneNodePos = m_grid->logicToScene(logicNodePos);

    QPointF leftLogicOffset = topLeft.data(ROLE_LEFTPOS).toPointF();
    QPointF rightLogicOffset = topLeft.data(ROLE_RIGHTPOS).toPointF();

    switch (role)
    {
        case ROLE_NODEPOS:
        {
            pNode->setPos(m_grid->logicToScene(logicNodePos));
            break;
        }
        case ROLE_LEFTPOS:
        {
            CurveHandlerItem* pLeftHdl = pNode->leftHandle();
            leftLogicOffset = pNode->mapFromScene(m_grid->logicToScene(leftLogicOffset + logicNodePos));
            pLeftHdl->setPos(leftLogicOffset);
            break;
        }
        case ROLE_RIGHTPOS:
        {
            CurveHandlerItem* pRightHdl = pNode->rightHandle();
            rightLogicOffset = pNode->mapFromScene(m_grid->logicToScene(rightLogicOffset + logicNodePos));
            pRightHdl->setPos(rightLogicOffset);
            break;
        }
        case ROLE_TYPE:
        {
            HANDLE_TYPE type = (HANDLE_TYPE)topLeft.data(ROLE_TYPE).toInt();
            CurveHandlerItem *pLeftHdl = pNode->leftHandle();
            CurveHandlerItem *pRightHdl = pNode->rightHandle();
            if (type == HDL_VECTOR)
            {
                if (pLeftHdl) pLeftHdl->toggle(false);
                if (pRightHdl) pRightHdl->toggle(false);
            }
            else
            {
                if (pLeftHdl && r > 0) pLeftHdl->toggle(true);
                if (pRightHdl && r < m_model->rowCount() - 1) pRightHdl->toggle(true);
            }
            break;
        }
    }

    QGraphicsPathItem* pLeftCurve = r > 0 ? m_vecCurves[r - 1] : nullptr;
    if (pLeftCurve)
	{
        int lType = topLeft.data(ROLE_TYPE).toInt();
        CurveNodeItem* pLeftNode = m_vecNodes[r - 1];
        QPainterPath path;
        path.moveTo(pLeftNode->pos());
        path.cubicTo(pLeftNode->rightHandlePos(), pNode->leftHandlePos(), pNode->pos());
        pLeftCurve->setPath(path);
        pLeftCurve->update();
	}

	QGraphicsPathItem* pRightCurve = (r < m_vecNodes.size() - 1) ? m_vecCurves[r] : nullptr;
    if (pRightCurve)
	{
        int rType = topLeft.data(ROLE_TYPE).toInt();
        CurveNodeItem* pRightNode = m_vecNodes[r + 1];
        QPainterPath path;
        path.moveTo(pNode->pos());
        path.cubicTo(pNode->rightHandlePos(), pRightNode->leftHandlePos(), pRightNode->pos());
        pRightCurve->setPath(path);
        pRightCurve->update();
	}
}

void CurvesItem::onNodesInserted(const QModelIndex& parent, int first, int last)
{
    //update curve
    for (int r = first; r <= last; r++)
    {
        const QModelIndex& idx = m_model->index(r, 0, parent);
        //can only insert into [1,n-1]
        ZASSERT_EXIT(r > 0 && r < m_vecNodes.size());

        QPointF pos = idx.data(ROLE_NODEPOS).toPointF();
        QPointF nodeScenePos = m_grid->logicToScene(pos);
        QPointF leftOffset = idx.data(ROLE_LEFTPOS).toPointF();
        QPointF rightOffset = idx.data(ROLE_RIGHTPOS).toPointF();

        //insert a new node.
        CurveNodeItem *pNewNode = new CurveNodeItem(idx, m_view, nodeScenePos, m_grid, this);
        connect(pNewNode, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));
        connect(pNewNode, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));

        QPointF leftScenePos = m_grid->logicToScene(pos + leftOffset);
        QPointF rightScenePos = m_grid->logicToScene(pos + rightOffset);
        QPointF leftSceneOffset = leftScenePos - nodeScenePos;
        QPointF rightSceneOffset = rightScenePos - nodeScenePos;

        pNewNode->initHandles(leftSceneOffset, rightSceneOffset);
        m_vecNodes.insert(r, pNewNode);

        // update curve
        CurveNodeItem *pLeftNode = m_vecNodes[r - 1];
        CurveNodeItem *pRightNode = m_vecNodes[r + 1];

        QPointF leftNodePos = pLeftNode->pos(),
                rightHdlPos = pLeftNode->rightHandlePos(),
                leftHdlPos = pRightNode->leftHandlePos(),
                rightNodePos = pRightNode->pos();

        CurvePathItem *pLeftHalf = m_vecCurves[r - 1];
        CurvePathItem *pRightHalf = new CurvePathItem(m_color, this);
        connect(pRightHalf, SIGNAL(clicked(const QPointF&)), this, SLOT(onPathClicked(const QPointF &)));
        m_vecCurves.insert(r, pRightHalf);

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
    }
}

void CurvesItem::onNodesAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    //sync to view.
    for (int i = first; i <= last; i++)
    {
        CurveNodeItem *pLeftNode = m_vecNodes[i - 1];
        CurveNodeItem *pRightNode = m_vecNodes[i + 1];

        //curves[i-1] as a new curve from node i-1 to node i.
        CurvePathItem *pathItem = m_vecCurves[i - 1];

        m_vecCurves[i]->deleteLater();
        m_vecNodes[i]->deleteLater();

        CurvePathItem *pDeleleCurve = m_vecCurves[i];
        m_vecCurves.remove(i);
        m_vecNodes.remove(i);

        QPainterPath path;
        path.moveTo(pLeftNode->pos());
        path.cubicTo(pLeftNode->rightHandlePos(), pRightNode->leftHandlePos(),
                     pRightNode->pos());
        pathItem->setPath(path);
        pathItem->update();
    }
}

int CurvesItem::indexOf(CurveNodeItem *pItem) const
{
    return m_vecNodes.indexOf(pItem);
}

void CurvesItem::setColor(const QColor& color)
{
    m_color = color;
    for (auto item : m_vecCurves)
    {
        const int penWidth = item->pen().width();
        QPen pen(color, penWidth);
        item->setPen(pen);
    }
}

void CurvesItem::_setVisible(bool bVisible)
{
    for (auto item : m_vecNodes) {
        item->setVisible(bVisible);
    }
    for (auto item : m_vecCurves) {
        item->setVisible(bVisible);
    }
}

int CurvesItem::nodeCount() const
{
    return m_vecNodes.size();
}

QPointF CurvesItem::nodePos(int i) const
{
    ZASSERT_EXIT(i >= 0 && i < m_vecNodes.size(), QPointF());
    return m_vecNodes[i]->pos();
}

CurveNodeItem* CurvesItem::nodeItem(int i) const
{
    ZASSERT_EXIT(i >= 0 && i < m_vecNodes.size(), nullptr);
    return m_vecNodes[i];
}

CurveModel* CurvesItem::model() const
{
    return m_model;
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
    ZASSERT_EXIT(i >= 0);

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
}

void CurvesItem::onNodeDeleted()
{
    CurveNodeItem* pItem = qobject_cast<CurveNodeItem*>(sender());
    ZASSERT_EXIT(pItem);
    int i = m_vecNodes.indexOf(pItem);
    if (i == 0 || i == m_vecNodes.size() - 1)
        return;

    m_model->removeRow(i);
}

void CurvesItem::onPathClicked(const QPointF& pos)
{
    CurvePathItem* pItem = qobject_cast<CurvePathItem*>(sender());
    ZASSERT_EXIT(pItem);
    int i = m_vecCurves.indexOf(pItem);

    CURVE_RANGE rg = m_model->range();
    qreal xscale = (rg.xTo - rg.xFrom) / 10.;

    QPointF logicPos = m_grid->sceneToLogic(pos);
    QPointF leftOffset(-xscale, 0);
    QPointF rightOffset(xscale, 0);

    QStandardItem* pModelItem = new QStandardItem;
    pModelItem->setData(logicPos, ROLE_NODEPOS);
    pModelItem->setData(leftOffset, ROLE_LEFTPOS);
    pModelItem->setData(rightOffset, ROLE_RIGHTPOS);
    pModelItem->setData(HDL_ASYM, ROLE_TYPE);
    m_model->insertRow(i + 1, pModelItem);
}