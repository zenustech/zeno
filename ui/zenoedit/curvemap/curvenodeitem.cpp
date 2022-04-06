#include "curvenodeitem.h"
#include "curvemapview.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/util/uihelper.h>
#include "curveutil.h"

using namespace curve_util;

CurveHandlerItem::CurveHandlerItem(CurveNodeItem* pNode, const QModelIndex& idx, const QPointF& offset, QGraphicsItem* parent)
	: QGraphicsRectItem(-3, -3, 6, 6, parent)
	, m_node(pNode)
	, m_index(idx)
	, m_nodeIdx(pNode->index())
{
	m_line = new QGraphicsLineItem(this);
	m_line->setPen(QPen(QColor(255,255,255), 1));
	m_line->setZValue(-100);

	setBrush(QColor(255, 87, 0));
	QPen pen;
	pen.setColor(QColor(255,255,255));
	pen.setWidth(1);
	setPen(pen);

	QPointF center = pNode->boundingRect().center();

	QPointF pos = offset;
	setPos(pos);

	m_line->setLine(QLineF(QPointF(0, 0), -offset));

	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
}

void CurveHandlerItem::setOtherHandleIdx(const QModelIndex& idx)
{
	m_otherIdx = idx;
}

QVariant CurveHandlerItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
	if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF nodePos = mapFromScene(m_node->scenePos());
		QPointF thisPos = boundingRect().center();
		m_line->setLine(QLineF(thisPos, nodePos));

		QPointF wtf = scenePos();
		pModel->setData(m_index, wtf, ROLE_ItemPos);
	}
	else if (change == QGraphicsItem::ItemSelectedChange)
	{
		bool isSelected = this->isSelected();
		int j;
		j = 0;
	}
	else if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		bool isSelected = this->isSelected();
		if (isSelected)
		{
			pModel->setData(m_index, ITEM_SELECTED, ROLE_ItemStatus);
			pModel->setData(m_nodeIdx, ITEM_TOGGLED, ROLE_ItemStatus);
			pModel->setData(m_otherIdx, ITEM_TOGGLED, ROLE_ItemStatus);
		}
		else
		{
			pModel->setData(m_index, ITEM_UNTOGGLED, ROLE_ItemStatus);
			pModel->setData(m_nodeIdx, ITEM_UNTOGGLED, ROLE_ItemStatus);
			if (!m_otherIdx.data(ROLE_MouseClicked).toBool())
				pModel->setData(m_otherIdx, ITEM_UNTOGGLED, ROLE_ItemStatus);
		}
	}
	return value;
}

void CurveHandlerItem::updateStatus()
{
	ItemStatus status = (ItemStatus)m_index.data(ROLE_ItemStatus).toInt();
	if (status == ITEM_SELECTED || status == ITEM_TOGGLED)
	{
		setVisible(true);
	}
	else
	{
		setVisible(false);
	}
}

void CurveHandlerItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
	pModel->setData(m_index, true, ROLE_MouseClicked);
	_base::mousePressEvent(event);
	pModel->setData(m_index, false, ROLE_MouseClicked);
}

void CurveHandlerItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	_base::mouseReleaseEvent(event);
}

void CurveHandlerItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	painter->setRenderHint(QPainter::Antialiasing);
	_base::paint(painter, option, widget);
}


CurveNodeItem::CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, QGraphicsItem* parentItem)
	: QGraphicsObject(parentItem)
	, m_left(nullptr)
	, m_right(nullptr)
	, m_view(pView)
	, m_bToggle(false)
{
	m_logicPos = m_view->mapSceneToLogic(nodePos);
	QRectF br = boundingRect();
	setPos(nodePos);
	setZValue(100);
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
}

void CurveNodeItem::initHandles(const MODEL_PACK& pack, const QModelIndex& idx, const QPointF& leftOffset, const QPointF& rightOffset)
{
	m_index = idx;
	const QString& nodeid = m_index.data(ROLE_ItemObjId).toString();
	QStandardItem* pLeftHandle = nullptr, *pRightHandle = nullptr;
	QModelIndex leftIdx, rightIdx;
	if (leftOffset != QPointF(0, 0))
	{
		pLeftHandle = new QStandardItem;
		pLeftHandle->setData(ITEM_LEFTHANDLE, ROLE_ItemType);
		pLeftHandle->setData(ITEM_UNTOGGLED, ROLE_ItemStatus);

		QPointF globalScenePos = scenePos() + leftOffset;
		pLeftHandle->setData(globalScenePos, ROLE_ItemPos);
		pLeftHandle->setData(UiHelper::generateUuid(), ROLE_ItemObjId);
		pLeftHandle->setData(nodeid, ROLE_ItemBelongTo);
		pack.pModel->appendRow(pLeftHandle);

		leftIdx = pack.pModel->indexFromItem(pLeftHandle);
		m_left = new CurveHandlerItem(this, leftIdx, leftOffset, this);
		m_left->hide();
	}

	if (rightOffset != QPointF(0, 0))
	{
		pRightHandle = new QStandardItem;
		pRightHandle->setData(ITEM_RIGHTHANDLE, ROLE_ItemType);
		pRightHandle->setData(ITEM_UNTOGGLED, ROLE_ItemStatus);

		QPointF globalScenePos = scenePos() + rightOffset;
		pRightHandle->setData(globalScenePos, ROLE_ItemPos);

		pRightHandle->setData(UiHelper::generateUuid(), ROLE_ItemObjId);
		pRightHandle->setData(nodeid, ROLE_ItemBelongTo);
		pack.pModel->appendRow(pRightHandle);

		rightIdx = pack.pModel->indexFromItem(pRightHandle);
		m_right = new CurveHandlerItem(this, rightIdx, rightOffset, this);
		m_right->hide();
	}

	if (m_left && m_right)
	{
		m_left->setOtherHandleIdx(rightIdx);
		m_right->setOtherHandleIdx(leftIdx);
	}
}

void CurveNodeItem::updateStatus()
{
	ItemStatus status = (ItemStatus)m_index.data(ROLE_ItemStatus).toInt();
	if (status == ITEM_SELECTED || status == ITEM_TOGGLED)
	{
		m_bToggle = true;
	}
	else
	{
		m_bToggle = false;
	}
	update();
}

void CurveNodeItem::updateHandleStatus(const QString& objId)
{
	if (m_left && m_left->index().data(ROLE_ItemObjId).toString() == objId)
		m_left->updateStatus();
	else if (m_right && m_right->index().data(ROLE_ItemObjId).toString() == objId)
		m_right->updateStatus();
}

QVariant CurveNodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		bool selected = isSelected();
		if (selected)
		{
			pModel->setData(m_index, ITEM_SELECTED, ROLE_ItemStatus);
			if (m_left)
				pModel->setData(m_left->index(), ITEM_TOGGLED, ROLE_ItemStatus);
			if (m_right)
				pModel->setData(m_right->index(), ITEM_TOGGLED, ROLE_ItemStatus);
		}
		else
		{
			pModel->setData(m_index, ITEM_UNTOGGLED, ROLE_ItemStatus);
			if (m_left && !m_left->index().data(ROLE_MouseClicked).toBool())
			{
				pModel->setData(m_left->index(), ITEM_UNTOGGLED, ROLE_ItemStatus);
			}
			if (m_right && !m_right->index().data(ROLE_MouseClicked).toBool())
			{
				pModel->setData(m_right->index(), ITEM_UNTOGGLED, ROLE_ItemStatus);
			}
		}
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF phyPos = scenePos();
		m_logicPos = m_view->mapSceneToLogic(phyPos);
		QPointF wtf = pos();
		pModel->setData(m_index, wtf, ROLE_ItemPos);
	}
	return value;
}

QPointF CurveNodeItem::logicPos() const
{
	return m_logicPos;
}

QRectF CurveNodeItem::boundingRect(void) const
{
	QSize sz = ZenoStyle::dpiScaledSize(QSize(20, 20));
	qreal w = sz.width(), h = sz.height();
	QRectF rc = QRectF(-w / 2, -h / 2, w, h);
	return rc;
}

void CurveNodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* opt, QWidget* widget)
{
	const QTransform& transform = m_view->transform();
	qreal scaleX = transform.m11();
	qreal scaleY = transform.m22();
	qreal width = opt->rect.width();
	qreal height = opt->rect.height();
	QPointF center = opt->rect.center();
	QRectF rc = opt->rect;
	bool bSelected = isSelected();
	painter->setBrush(QColor(0, 0, 0));
	//painter->drawRect(rc);

	qreal W = 0, H = 0;
	if (scaleX < scaleY)
	{
		W = width;
		H = width * scaleX / scaleY;
	}
	else
	{
		W = height * scaleY / scaleX;
		H = height;
	}

	if (m_bToggle)
	{
		painter->setPen(QPen(QColor(255,255,255), 1));
		painter->setBrush(QColor(255, 87, 0));
		painter->setRenderHint(QPainter::Antialiasing, true);

		QPainterPath path;
		W = 0.8 * W;
		H = 0.8 * H;
		path.moveTo(center.x(), center.y() - H / 2);
		path.lineTo(center.x() + W / 2, center.y());
		path.lineTo(center.x(), center.y() + H / 2);
		path.lineTo(center.x() - W / 2, center.y());
		path.closeSubpath();

		painter->drawPath(path);
	}
	else
	{
		painter->setRenderHint(QPainter::Antialiasing, true);
		painter->setPen(Qt::NoPen);
		painter->setBrush(QColor(231, 29, 31));
		painter->drawEllipse(center, W / 4, H / 4);
	}
}