#include "blackboardnode.h"
#include <zenoui/render/common_id.h>


BlackboardNode::BlackboardNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent)
    , m_bDragging(false)
    , m_pTextItem(nullptr)
{
    m_ptBottomRight = QPointF(256, 180);
}

BlackboardNode::~BlackboardNode()
{
}

QRectF BlackboardNode::boundingRect() const
{
    return ZenoNode::boundingRect();
    //QRectF rc = QRectF(QPointF(0, 0), m_ptBottomRight);
    //return rc;
}

ZenoBackgroundWidget* BlackboardNode::initBodyWidget(NODE_TYPE type)
{
    ZenoBackgroundWidget *bodyWidget = new ZenoBackgroundWidget(this);

    bodyWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal, bodyBg.clr_hovered, bodyBg.clr_selected);
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);
    qreal border = m_renderParams.bodyBg.border_witdh;
    pVLayout->setContentsMargins(border, border, border, border);

    BLACKBOARD_INFO blackboard = index().data(ROLE_BLACKBOARD).value<BLACKBOARD_INFO>();
    ZenoParamBlackboard* pTextEdit = new ZenoParamBlackboard(blackboard.content, m_renderParams.lineEditParam);
    pVLayout->addItem(pTextEdit);

    bodyWidget->setLayout(pVLayout);
    return bodyWidget;
}

ZenoBackgroundWidget* BlackboardNode::initHeaderWangStyle(NODE_TYPE type)
{
    ZenoBackgroundWidget *headerWidget = new ZenoBackgroundWidget(this);
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
    headerWidget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);
    headerWidget->setBorder(headerBg.border_witdh, headerBg.clr_border);

    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoSpacerItem *pSpacerItem = new ZenoSpacerItem(true, 100);

    const QString &name = index().data(ROLE_OBJNAME).toString();
    auto nameItem = new ZenoTextLayoutItem(name, m_renderParams.nameFont, m_renderParams.nameClr.color(), this);
    QGraphicsLinearLayout *pNameLayout = new QGraphicsLinearLayout(Qt::Horizontal);
    pNameLayout->addItem(nameItem);
    pNameLayout->setContentsMargins(5, 5, 5, 5);

    int options = index().data(ROLE_OPTIONS).toInt();

    pHLayout->addItem(pNameLayout);
    pHLayout->addItem(pSpacerItem);
    pHLayout->setSpacing(0);
    pHLayout->setContentsMargins(0, 0, 0, 0);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    headerWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

    if (type == BLACKBOARD_NODE) {
        QColor clr(98, 108, 111);
        headerWidget->setColors(false, clr, clr, clr);
    }
    return headerWidget;
}

void BlackboardNode::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoNode::mousePressEvent(event);

    QPointF pos = event->pos();
    if (isDragArea(pos)) {
        m_bDragging = true;
    }
    else {
        m_bDragging = false;
        //m_ptBottomRight = pos;
    }
}

void BlackboardNode::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_bDragging)
    {
        QPointF topLeft = m_bodyWidget->sceneBoundingRect().topLeft();
        QPointF newPos = event->scenePos();
        QPointF currPos = m_bodyWidget->sceneBoundingRect().bottomRight();

        qreal newWidth = newPos.x() - topLeft.x();
        qreal newHeight = newPos.y() - topLeft.y() + m_headerWidget->size().height();

        QSizeF oldSz = this->size();

        resize(QSizeF(newWidth, newHeight));
        //updateWhole();
        return;
    }

    ZenoNode::mouseMoveEvent(event);
}

void BlackboardNode::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_bDragging) {
        m_bDragging = false;
        updateWhole();
        return;
    }
    ZenoNode::mouseReleaseEvent(event);
}

void BlackboardNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    ZenoNode::mouseDoubleClickEvent(event);
    //if (m_pTextItem) {
    //    m_pTextItem->setTextInteractionFlags(Qt::TextEditorInteraction);
    //}
}

void BlackboardNode::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    ZenoNode::hoverEnterEvent(event);
}

void BlackboardNode::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    ZenoNode::hoverLeaveEvent(event);
}

void BlackboardNode::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    ZenoNode::hoverMoveEvent(event);
    bool bDrag = isDragArea(event->pos());
    if (bDrag) {
        setCursor(QCursor(Qt::SizeFDiagCursor));
    } else {
        setCursor(QCursor(Qt::ArrowCursor));
    }
}

bool BlackboardNode::isDragArea(QPointF pos)
{
    QPointF bottomright = boundingRect().bottomRight();
    QPointF offset = pos - bottomright;
    return (offset.manhattanLength() < 10);
}
