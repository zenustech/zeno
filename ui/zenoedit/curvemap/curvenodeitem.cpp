#include "curvenodeitem.h"
#include "curvemapview.h"
#include <zenoui/style/zenostyle.h>
#include <zenoui/util/uihelper.h>
#include "curveutil.h"

using namespace curve_util;


CurvePathItem::CurvePathItem(QGraphicsItem *parent)
    : QObject(nullptr)
	, QGraphicsPathItem(parent)
{
    const int penWidth = 2;
    QPen pen(QColor(231, 29, 31), penWidth);
    pen.setStyle(Qt::SolidLine);
    setPen(pen);
}

void CurvePathItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QGraphicsPathItem::mousePressEvent(event);
    emit clicked(event->pos());
}


CurveHandlerItem::CurveHandlerItem(CurveNodeItem* pNode, const QPointF& offset, QGraphicsItem* parent)
	: QGraphicsRectItem(-3, -3, 6, 6, parent)
	, m_node(pNode)
	, m_bMouseTriggered(false)
	, m_other(nullptr)
	, m_bNotify(true)
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
    setFlag(ItemSendsGeometryChanges, true);
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
            int i = m_node->grid()->indexOf(m_node);
            if (m_node->leftHandle() == this)
			{
                QPointF lastNodePos = m_node->mapFromScene(m_node->grid()->nodePos(i - 1));
                newPos.setX(qMax(lastNodePos.x(), newPos.x()));
                if (m_other)
				{
                    QPointF rightHdlPos = m_other->pos();
                    newPos.setX(qMin(newPos.x(), rightHdlPos.x()));
				}
            }
			else if (m_node->rightHandle() == this)
			{
                QPointF nextNodePos = m_node->mapFromScene(m_node->grid()->nodePos(i + 1));
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
		QPointF nodePos = mapFromScene(m_node->scenePos());
		QPointF thisPos = boundingRect().center();
		m_line->setLine(QLineF(thisPos, nodePos));

		QPointF wtf = scenePos();
        if (m_bNotify)
			m_node->onHandleUpdate(this);
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
				setVisible(true);
				if (m_other)
					m_other->setVisible(true);
			}
			else
			{
				m_node->toggle(false);
				setVisible(false);
				if (m_other && !m_other->isMouseEventTriggered())
					m_other->setVisible(false);
			}
		}
	}
	return value;
}

bool CurveHandlerItem::isMouseEventTriggered()
{
    return m_bMouseTriggered;
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

void CurveHandlerItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	painter->setRenderHint(QPainter::Antialiasing);
	_base::paint(painter, option, widget);
}


CurveNodeItem::CurveNodeItem(CurveMapView* pView, const QPointF& nodePos, CurveGrid* parentItem)
	: QGraphicsObject(parentItem)
	, m_left(nullptr)
	, m_right(nullptr)
	, m_view(pView)
	, m_bToggle(false)
	, m_grid(parentItem)
{
    QRectF br = boundingRect();
	setPos(nodePos);
	setZValue(100);
	setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges | ItemIsFocusable);
}

void CurveNodeItem::initHandles(const QPointF& leftOffset, const QPointF& rightOffset)
{
	if (leftOffset != QPointF(0, 0))
	{
		m_left = new CurveHandlerItem(this, leftOffset, this);
		m_left->hide();
	}

	if (rightOffset != QPointF(0, 0))
	{
		m_right = new CurveHandlerItem(this, rightOffset, this);
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
    if (m_view->isSmoothCurve())
	{
		//update the pos of another handle.
        QVector2D roffset(rightHandlePos() - pos());
        QVector2D loffset(leftHandlePos() - pos());
        if (m_left == pItem && m_right)
		{
			qreal length = roffset.length();
			roffset = -loffset.normalized() * length;
			QPointF newPos = roffset.toPointF() + pos();
			newPos = mapFromScene(newPos);

			m_right->setUpdateNotify(false);
			m_right->setPos(newPos);
			m_right->setUpdateNotify(true);
		}
		else if (m_right == pItem && m_left)
		{
			qreal length = loffset.length();
			loffset = -roffset.normalized() * length;
			QPointF newPos = loffset.toPointF() + pos();
			newPos = mapFromScene(newPos);

			m_left->setUpdateNotify(false);
			m_left->setPos(newPos);
			m_left->setUpdateNotify(true);
		}
    }
	emit geometryChanged();
}

void CurveNodeItem::toggle(bool bChecked)
{
    m_bToggle = bChecked;
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

QVariant CurveNodeItem::itemChange(GraphicsItemChange change, const QVariant& value)
{
	if (change == QGraphicsItem::ItemSelectedHasChanged)
	{
		bool selected = isSelected();
		if (selected)
		{
            m_bToggle = true;
			if (m_left) m_left->setVisible(true);
			if (m_right) m_right->setVisible(true);
		}
		else
		{
			m_bToggle = false;
			if (m_left && !m_left->isMouseEventTriggered())
			{
				m_left->setVisible(false);
			}
			if (m_right && !m_right->isMouseEventTriggered())
			{
				m_right->setVisible(false);
			}
		}
	} 
	else if (change == QGraphicsItem::ItemPositionChange)
	{
        int i = m_grid->indexOf(this);

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
			else if (i == grid()->nodeCount() - 1)
			{
                newPos.setX(rc.right());
            }
			else
			{
                CurveNodeItem *pLast = m_grid->nodeItem(i - 1);
                CurveNodeItem *pNext = m_grid->nodeItem(i + 1);
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
		emit geometryChanged();
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