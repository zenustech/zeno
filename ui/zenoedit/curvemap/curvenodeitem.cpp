#include "curvenodeitem.h"
#include "curvemapview.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/util/uihelper.h>
#include "curveutil.h"
#include "curvesitem.h"
#include "../model/curvemodel.h"

using namespace curve_util;


CurvePathItem::CurvePathItem(QGraphicsItem *parent)
    : QObject(nullptr)
	, QGraphicsPathItem(parent)
{
    const int penWidth = 2;
    QPen pen(QColor(77, 77, 77), penWidth);
    pen.setStyle(Qt::SolidLine);
    setPen(pen);
    setZValue(9);
}

void CurvePathItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsPathItem::mousePressEvent(event);
    emit clicked(event->pos());
}


CurveHandlerItem::CurveHandlerItem(CurveNodeItem* pNode, const QPointF& offset, QGraphicsItem* parent)
    : _base(parent)
	, m_node(pNode)
	, m_bMouseTriggered(false)
	, m_other(nullptr)
	, m_bNotify(true)
{
    CurveGrid *pGrid = m_node->grid();

    m_line = new QGraphicsLineItem(pGrid);
	m_line->setPen(QPen(QColor(255, 255, 255), 2));

	QPointF center = pNode->boundingRect().center();

	QPointF pos = offset;
	setPos(pos);

	QPointF wtf = this->scenePos();
	QPointF hdlPosInGrid = pGrid->mapFromScene(wtf);
    QPointF nodePosInGrid = pNode->pos();

    m_line->setLine(QLineF(hdlPosInGrid, nodePosInGrid));
    m_line->setZValue(10);

	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);
    setFlag(ItemSendsGeometryChanges, true);
}

CurveHandlerItem::~CurveHandlerItem()
{
    delete m_line;
}

void CurveHandlerItem::setOtherHandle(CurveHandlerItem *other)
{
    m_other = other;
}

QVariant CurveHandlerItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
    if (change == QGraphicsItem::ItemPositionChange)
	{
        if (m_node->grid()->isFuncCurve())
		{
            QPointF newPos = value.toPointF();
            int i = m_node->curves()->indexOf(m_node);
            if (m_node->leftHandle() == this)
			{
                QPointF lastNodePos = m_node->mapFromScene(m_node->curves()->nodePos(i - 1));
                newPos.setX(qMax(lastNodePos.x(), newPos.x()));
                if (m_other)
				{
                    QPointF rightHdlPos = m_other->pos();
                    newPos.setX(qMin(newPos.x(), rightHdlPos.x()));
				}
            }
			else if (m_node->rightHandle() == this)
			{
                QPointF nextNodePos = m_node->mapFromScene(m_node->curves()->nodePos(i + 1));
                newPos.setX(qMin(nextNodePos.x(), newPos.x()));
                if (m_other)
				{
                    QPointF leftHdlPos = m_other->pos();
                    newPos.setX(qMax(newPos.x(), leftHdlPos.x()));
				}
            }
            return newPos;
		}
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
		QPointF hdlPosInGrid = m_node->grid()->mapFromScene(scenePos());
		QPointF nodePosInGrid = m_node->pos();
		m_line->setLine(QLineF(hdlPosInGrid, nodePosInGrid));

        if (m_bNotify)
			m_node->onHandleUpdate(this);
	}
	else if (change == QGraphicsItem::ItemScenePositionHasChanged)
	{
		QPointF hdlPosInGrid = m_node->grid()->mapFromScene(scenePos());
		QPointF nodePosInGrid = m_node->pos();
		m_line->setLine(QLineF(hdlPosInGrid, nodePosInGrid));
	}
	else if (change == QGraphicsItem::ItemSelectedChange)
	{
		bool isSelected = this->isSelected();
		int j;
		j = 0;
	}
	else if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		if (m_bNotify)
		{
            bool isSelected = this->isSelected();
			if (isSelected)
			{
				m_node->toggle(true);
				toggle(true);
				if (m_other)
					m_other->toggle(true);
			}
			else
			{
				m_node->toggle(false);
				toggle(false);
				if (m_other && !m_other->isMouseEventTriggered())
					m_other->toggle(false);
			}
		}
	}
	return value;
}

bool CurveHandlerItem::isMouseEventTriggered()
{
    return m_bMouseTriggered;
}

void CurveHandlerItem::toggle(bool bToggle)
{
    if (pos() == QPointF(0, 0))
	{
        bToggle = false;
	}
    setVisible(bToggle);
    m_line->setVisible(bToggle);
}

int CurveHandlerItem::type() const
{
	return Type;
}

CurveNodeItem* CurveHandlerItem::nodeItem() const
{
    return m_node;
}

void CurveHandlerItem::setUpdateNotify(bool bNotify)
{
    m_bNotify = bNotify;
}

void CurveHandlerItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    m_bMouseTriggered = true;
	_base::mousePressEvent(event);
    m_bMouseTriggered = false;
}

void CurveHandlerItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    m_bMouseTriggered = true;
	_base::mouseReleaseEvent(event);
    m_bMouseTriggered = false;
}

QRectF CurveHandlerItem::boundingRect(void) const
{
    QSize sz = ZenoStyle::dpiScaledSize(QSize(12, 12));
    qreal w = sz.width(), h = sz.height();
    QRectF rc = QRectF(-w / 2, -h / 2, w, h);
    return rc;
}

void CurveHandlerItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	const QTransform &transform = m_node->grid()->view()->transform();
	qreal scaleX = transform.m11();
	qreal scaleY = transform.m22();
	qreal width = option->rect.width();
	qreal height = option->rect.height();
	QPointF center = option->rect.center();
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

	W = 0.5 * W;
	H = 0.5 * H;

    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(255, 255, 255));
    painter->drawEllipse(center, W, H);
}


CurveNodeItem::CurveNodeItem(const QModelIndex& idx, CurveMapView* pView, const QPointF& nodePos, CurveGrid* parentItem, CurvesItem* curve)
	: QGraphicsObject(parentItem)
	, m_left(nullptr)
	, m_right(nullptr)
	, m_view(pView)
	, m_bToggle(false)
	, m_grid(parentItem)
	, m_type(HDL_ASYM)
	, m_curve(curve)
	, m_index(idx)
{
    QRectF br = boundingRect();
	setPos(nodePos);
    setZValue(11);
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges | ItemIsFocusable);
}

void CurveNodeItem::initHandles(const QPointF& leftOffset, const QPointF& rightOffset)
{
	m_left = new CurveHandlerItem(this, leftOffset, this);
	if (leftOffset == QPointF(0, 0))
	{
		m_left->hide();
	}

	m_right = new CurveHandlerItem(this, rightOffset, this);
	if (rightOffset == QPointF(0, 0))
	{
		m_right->hide();
	}

	if (m_left && m_right)
	{
		m_left->setOtherHandle(m_right);
        m_right->setOtherHandle(m_left);
	}
}

void CurveNodeItem::onHandleUpdate(CurveHandlerItem* pItem)
{
	Q_ASSERT(pItem && m_index.isValid());
	int type = m_index.data(ROLE_TYPE).toInt();
	{
		//update the pos of another handle.
        CurveModel *const pModel = m_curve->model();
        QVector2D roffset(rightHandlePos() - pos());
        QVector2D loffset(leftHandlePos() - pos());
        if (m_left == pItem)
		{
			if (m_right && type != HDL_FREE)
			{
				qreal length = roffset.length();
				if (type == HDL_ALIGNED)
					length = loffset.length();
				roffset = -loffset.normalized() * length;
				QPointF newPos = roffset.toPointF() + pos();
				newPos = mapFromScene(newPos);

				m_right->setUpdateNotify(false);
				m_right->setPos(newPos);
				m_right->setUpdateNotify(true);

				//sync to model.
				QPointF offset = m_grid->sceneToLogic(m_right->scenePos()) - m_grid->sceneToLogic(scenePos());
				pModel->setData(m_index, offset, ROLE_RIGHTPOS);
			}
			QPointF offset = m_grid->sceneToLogic(m_left->scenePos()) - m_grid->sceneToLogic(scenePos());
			pModel->setData(m_index, offset, ROLE_LEFTPOS);
		}
		
		if (m_right == pItem)
		{
			if (m_left && type != HDL_FREE)
			{
				qreal length = loffset.length();
				if (type == HDL_ALIGNED)
					length = roffset.length();
				loffset = -roffset.normalized() * length;
				QPointF newPos = loffset.toPointF() + pos();
				newPos = mapFromScene(newPos);

				m_left->setUpdateNotify(false);
				m_left->setPos(newPos);
				m_left->setUpdateNotify(true);

				//sync to model.
				QPointF offset = m_grid->sceneToLogic(m_left->scenePos()) - m_grid->sceneToLogic(scenePos());
				pModel->setData(m_index, offset, ROLE_LEFTPOS);
			}
			QPointF offset = m_grid->sceneToLogic(m_right->scenePos()) - m_grid->sceneToLogic(scenePos());
			pModel->setData(m_index, offset, ROLE_RIGHTPOS);
		}
    }
}

void CurveNodeItem::toggle(bool bChecked)
{
    m_bToggle = bChecked;
}

bool CurveNodeItem::isToggled() const
{
    return m_bToggle;
}

CurveHandlerItem* CurveNodeItem::leftHandle() const
{
    return m_left;
}

CurveHandlerItem* CurveNodeItem::rightHandle() const
{
    return m_right;
}

QPointF CurveNodeItem::leftHandlePos() const
{
    return m_left ? m_left->scenePos() : QPointF();
}

QPointF CurveNodeItem::rightHandlePos() const
{
    return m_right ? m_right->scenePos() : QPointF();
}

CurveGrid* CurveNodeItem::grid() const
{
    return m_grid;
}

CurvesItem* CurveNodeItem::curves() const
{
    return m_curve;
}

QModelIndex CurveNodeItem::index() const
{
	return m_index;
}

int CurveNodeItem::type() const
{
    return Type;
}

HANDLE_TYPE CurveNodeItem::hdlType() const
{
    return m_type;
}

void CurveNodeItem::setHdlType(HANDLE_TYPE type)
{
    m_type = type;
}

QVariant CurveNodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		bool selected = isSelected();
		if (selected)
		{
            toggle(true);
			if (m_index.data(ROLE_TYPE).toInt() != HDL_VECTOR)
			{
				if (m_left) m_left->toggle(true);
				if (m_right) m_right->toggle(true);
			}
			else
			{
				//handler length is 0.
				if (m_left) m_left->toggle(false);
				if (m_right) m_right->toggle(false);
			}
		}
		else
		{
			toggle(false);
			if (m_left && !m_left->isMouseEventTriggered())
			{
				m_left->toggle(false);
			}
			if (m_right && !m_right->isMouseEventTriggered())
			{
				m_right->toggle(false);
			}
		}
	} 
	else if (change == QGraphicsItem::ItemPositionChange)
	{
        int i = m_curve->indexOf(this);

		QRectF rc = m_grid->boundingRect();
		QPointF newPos = value.toPointF();
        newPos.setX(qMin(qMax(newPos.x(), rc.left()), rc.right()));
        newPos.setY(qMin(qMax(newPos.y(), rc.top()), rc.bottom()));

        if (grid()->isFuncCurve())
		{
            if (i == 0)
			{
                newPos.setX(rc.left());
			}
			else if (i == curves()->nodeCount() - 1)
			{
                newPos.setX(rc.right());
            }
			else
			{
                CurveNodeItem *pLast = m_curve->nodeItem(i - 1);
                CurveNodeItem *pNext = m_curve->nodeItem(i + 1);
                CurveHandlerItem *pRightHdl = pLast->rightHandle();
                CurveHandlerItem *pLeftHdl = pNext->leftHandle();
                newPos.setX(qMin(qMax(newPos.x(), pRightHdl->scenePos().x()),
                                 pLeftHdl->scenePos().x()));
			}
			return newPos;
        }
	}
	else if (change == QGraphicsItem::ItemPositionHasChanged)
	{
        CurveModel *const pModel = m_curve->model();
		QPointF logicPos = m_grid->sceneToLogic(pos());
		pModel->setData(m_index, logicPos, ROLE_NODEPOS);
	}
	return value;
}

QRectF CurveNodeItem::boundingRect(void) const
{
	QSize sz = ZenoStyle::dpiScaledSize(QSize(20, 20));
	qreal w = sz.width(), h = sz.height();
	QRectF rc = QRectF(-w / 2, -h / 2, w, h);
	return rc;
}

void CurveNodeItem::keyPressEvent(QKeyEvent* event)
{
    QGraphicsObject::keyPressEvent(event);
    if (event->key() == Qt::Key_Delete)
	{
        emit deleteTriggered();
	}
}

void CurveNodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* opt, QWidget* widget)
{
	const QTransform& transform = m_view->transform();
	qreal scaleX = transform.m11();
	qreal scaleY = transform.m22();
	qreal width = opt->rect.width();
	qreal height = opt->rect.height();
	QPointF center = opt->rect.center();

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
        QPen pen(QColor(245, 172, 83), 2);
		pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);
        painter->setBrush(QColor(245, 172, 83));
	}
    else
	{
        QPen pen(QColor(255, 255, 255), 2);
		pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);
        painter->setBrush(QColor(24, 24, 24)); 
	}

	QPainterPath path;
	W = 0.7 * W;
	H = 0.7 * H;
	path.moveTo(center.x(), center.y() - H / 2);
	path.lineTo(center.x() + W / 2, center.y());
	path.lineTo(center.x(), center.y() + H / 2);
	path.lineTo(center.x() - W / 2, center.y());
	path.closeSubpath();

	painter->drawPath(path);
}