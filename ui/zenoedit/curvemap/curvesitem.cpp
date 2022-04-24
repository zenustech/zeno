#include "curvegrid.h"
#include "curvenodeitem.h"
#include "curvemapview.h"
#include "curvesitem.h"
#include "../model/curvemodel.h"
#include <zenoui/util/uihelper.h>


CurvesItem::CurvesItem(CurveMapView* pView, CurveGrid* grid, const QRectF& rc, QGraphicsItem* parent)
    : QGraphicsObject(parent)
    , m_view(pView)
    , m_grid(grid)
    , m_model(nullptr)
{
}

CurvesItem::~CurvesItem()
{
    //todo: delete all nodes and curves.
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

		CurvePathItem* pathItem = new CurvePathItem(this);
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
    Q_ASSERT(r >= 0 && r < m_vecNodes.size());
    CurveNodeItem *pNode = m_vecNodes[r];

    Q_ASSERT(!roles.isEmpty());
    int role = roles[0];
    QPointF logicNodePos = topLeft.data(ROLE_NODEPOS).toPointF();
    QPointF sceneNodePos = m_grid->logicToScene(logicNodePos);
    HANDLE_TYPE nodeType = (HANDLE_TYPE)topLeft.data(ROLE_TYPE).toInt();

    QPointF leftLogicOffset = topLeft.data(ROLE_LEFTPOS).toPointF();
    QPointF rightLogicOffset = topLeft.data(ROLE_RIGHTPOS).toPointF();
    QVector2D roffset(m_grid->logicToScene(rightLogicOffset + logicNodePos) - sceneNodePos);
    QVector2D loffset(m_grid->logicToScene(leftLogicOffset + logicNodePos) - sceneNodePos);

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
            pLeftHdl->setUpdateNotify(false);
            leftLogicOffset = pNode->mapFromScene(m_grid->logicToScene(leftLogicOffset + logicNodePos));
            pLeftHdl->setPos(leftLogicOffset);
            pLeftHdl->setUpdateNotify(true);

            //update another hdl.
            if (nodeType != HDL_FREE && rightLogicOffset != QPointF(0, 0))
            {
                qreal length = roffset.length();
                if (nodeType == HDL_ALIGNED)
                    length = loffset.length();
                roffset = -loffset.normalized() * length;
                QPointF newScenePos = roffset.toPointF() + sceneNodePos;
                QPointF newPos = pNode->mapFromScene(newScenePos);

                CurveHandlerItem *pRightHdl = pNode->rightHandle();
                pRightHdl->setUpdateNotify(false);
                pRightHdl->setPos(newPos);
                pRightHdl->setUpdateNotify(true);

                //have to manually update model data to avoid infinte notify loop.
                BlockSignalScope scope(m_model);
                rightLogicOffset = m_grid->sceneToLogic(newScenePos) - logicNodePos;
                m_model->setData(topLeft, rightLogicOffset, ROLE_RIGHTPOS);
            }
            break;
        }
        case ROLE_RIGHTPOS:
        {
            CurveHandlerItem* pRightHdl = pNode->rightHandle();
            rightLogicOffset = pNode->mapFromScene(m_grid->logicToScene(rightLogicOffset + logicNodePos));
            pRightHdl->setUpdateNotify(false);
            pRightHdl->setPos(rightLogicOffset);
            pRightHdl->setUpdateNotify(true);

            //update another hdl.
            if (nodeType != HDL_FREE && leftLogicOffset != QPointF(0, 0))
            {
                qreal length = loffset.length();
                if (nodeType == HDL_ALIGNED)
                    length = roffset.length();
                loffset = -roffset.normalized() * length;

                QPointF newScenePos = loffset.toPointF() + sceneNodePos;
                QPointF newPos = pNode->mapFromScene(newScenePos);

                CurveHandlerItem* pLeftHdl = pNode->leftHandle();
                pLeftHdl->setUpdateNotify(false);
                pLeftHdl->setPos(newPos);
                pLeftHdl->setUpdateNotify(true);

                BlockSignalScope scope(m_model);
                leftLogicOffset = m_grid->sceneToLogic(newScenePos) - logicNodePos;
                m_model->setData(topLeft, leftLogicOffset, ROLE_LEFTPOS);
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
        Q_ASSERT(r > 0 && r < m_vecNodes.size());

        QPointF pos = idx.data(ROLE_NODEPOS).toPointF();
        QPointF leftOffset = idx.data(ROLE_LEFTPOS).toPointF();
        QPointF rightOffset = idx.data(ROLE_RIGHTPOS).toPointF();

        //insert a new node.
        CurveNodeItem *pNewNode = new CurveNodeItem(idx, m_view, m_grid->logicToScene(pos), m_grid, this);
        connect(pNewNode, SIGNAL(geometryChanged()), this, SLOT(onNodeGeometryChanged()));
        connect(pNewNode, SIGNAL(deleteTriggered()), this, SLOT(onNodeDeleted()));
        pNewNode->initHandles(m_grid->logicToScene(leftOffset), m_grid->logicToScene(rightOffset));
        m_vecNodes.insert(r, pNewNode);

        // update curve
        CurveNodeItem *pLeftNode = m_vecNodes[r - 1];
        CurveNodeItem *pRightNode = m_vecNodes[r + 1];

        QPointF leftNodePos = pLeftNode->pos(),
                rightHdlPos = pLeftNode->rightHandlePos(),
                leftHdlPos = pRightNode->leftHandlePos(),
                rightNodePos = pRightNode->pos();

        CurvePathItem *pLeftHalf = m_vecCurves[r - 1];
        CurvePathItem *pRightHalf = new CurvePathItem(this);
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
}

void CurvesItem::onNodeDeleted()
{
    CurveNodeItem* pItem = qobject_cast<CurveNodeItem*>(sender());
    Q_ASSERT(pItem);
    int i = m_vecNodes.indexOf(pItem);
    if (i == 0 || i == m_vecNodes.size() - 1)
        return;

    m_model->removeRow(i);
}

void CurvesItem::onPathClicked(const QPointF& pos)
{
    CurvePathItem* pItem = qobject_cast<CurvePathItem*>(sender());
    Q_ASSERT(pItem);
    int i = m_vecCurves.indexOf(pItem);

    QPointF logicPos = m_grid->sceneToLogic(pos);
    QPointF leftOffset(-50, 0);
    QPointF rightOffset(50, 0);

    QStandardItem* pModelItem = new QStandardItem;
    pModelItem->setData(logicPos, ROLE_NODEPOS);
    pModelItem->setData(m_grid->sceneToLogic(leftOffset), ROLE_LEFTPOS);
    pModelItem->setData(m_grid->sceneToLogic(rightOffset), ROLE_RIGHTPOS);
    pModelItem->setData(HDL_ASYM, ROLE_TYPE);
    m_model->insertRow(i + 1, pModelItem);
}