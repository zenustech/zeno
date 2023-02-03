#include "blackboardnode2.h"
#include "util/log.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "zenoui/style/zenostyle.h"
#include <QPainter>
#include <zenomodel/include/graphsmanagment.h>
#include <zenomodel/include/uihelper.h>
#include <zenoui/comctrl/gv/zitemfactory.h>
#include <zenoui/render/common_id.h>

BlackboardNode2::BlackboardNode2(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent), 
    m_bDragging(false)
{
    setAutoFillBackground(false);
    setAcceptHoverEvents(true);
    resize(500, 500);
    onZoomed();
}

BlackboardNode2::~BlackboardNode2() {
}

void BlackboardNode2::updateClidItem(bool isAdd, const QString nodeId)
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (isAdd && !info.items.contains(nodeId)) {
        info.items << nodeId;
    } else if (!isAdd && info.items.contains(nodeId)) {
        info.items.removeOne(nodeId);
    }
    else {
        return;
    }
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), info, subGraphIndex(), true);
}

bool BlackboardNode2::nodePosChanged(ZenoNode *item) 
{
    if (this->sceneBoundingRect().contains(item->sceneBoundingRect()) && !m_childItems.contains(item)) {

        BlackboardNode2 *pParentItem = item->getGroupNode();
        if (pParentItem && pParentItem->sceneBoundingRect().contains(item->sceneBoundingRect()) &&
            (!pParentItem->sceneBoundingRect().contains(this->sceneBoundingRect()))) {
            return false;
        }
        if (pParentItem) {
            pParentItem->removeChildItem(item);
        }
        m_childItems << item;
        item->setGroupNode(this);
        if (item->zValue() <= zValue()) {
            item->setZValue(zValue() + 1);
         }
        updateClidItem(true, item->nodeId());
        return true;
    } else if ((!this->sceneBoundingRect().contains(item->sceneBoundingRect())) && m_childItems.contains(item)) {
        BlackboardNode2 *pGroupNode = this->getGroupNode();
        while (pGroupNode) {
            bool isUpdate = pGroupNode->nodePosChanged(item);
            if (!isUpdate) {
                pGroupNode = pGroupNode->getGroupNode();
            }
            else
                return true;
        }
        m_childItems.removeOne(item);
        item->setGroupNode(nullptr);
        updateClidItem(false, item->nodeId());
    }
    return false;
}

void BlackboardNode2::onZoomed() 
{
    int fontSize = 12 / editor_factor > 12 ? 12 / editor_factor : 12;
    QFont font("HarmonyOS Sans", fontSize);
    this->setFont(font);
}

QRectF BlackboardNode2::boundingRect() const {
    return ZenoNode::boundingRect();
}

void BlackboardNode2::onUpdateParamsNotDesc() 
{
    update();
}

void BlackboardNode2::appendChildItem(ZenoNode *item)
{
    m_childItems << item;
    for (auto item : m_childItems) {
        if (item->zValue() <= zValue()) {
            item->setZValue(zValue() + 1);
        }
    }
}

void BlackboardNode2::updateChildItemsPos()
{
    for (auto item : m_childItems) {
        if (!item) {
            continue;
        }
        int type = item->index().data(ROLE_NODETYPE).toInt();
        if (type == BLACKBOARD_NODE) 
        {
            BlackboardNode2 *pNode = dynamic_cast<BlackboardNode2 *>(item);
            if (pNode) {
                pNode->updateChildItemsPos();
            }
        }
        item->updateNodePos(item->pos());
    }
}

QVector<ZenoNode *> BlackboardNode2::getChildItems() 
{
    return m_childItems;
}

void BlackboardNode2::removeChildItem(ZenoNode *pNode) 
{
    if (m_childItems.contains(pNode)) {
        m_childItems.removeOne(pNode);
    }
}

ZLayoutBackground *BlackboardNode2::initBodyWidget(ZenoSubGraphScene *pScene) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
    }
    return new ZLayoutBackground(this);
}

ZLayoutBackground *BlackboardNode2::initHeaderWidget(IGraphsModel*)
{
    return new ZLayoutBackground(this);    
}

void BlackboardNode2::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    ZenoNode::mousePressEvent(event);

    QPointF pos = event->pos();
    if (isDragArea(pos)) {
        m_bDragging = true;
    } else {
        m_bDragging = false;
    }
}

void BlackboardNode2::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        QPointF topLeft = sceneBoundingRect().topLeft();
        QPointF newPos = event->scenePos();
        QPointF currPos = sceneBoundingRect().bottomRight();

        qreal newWidth = newPos.x() - topLeft.x();
        qreal newHeight = newPos.y() - topLeft.y();
        resize(newWidth, newHeight);
        return;
    }

    ZenoNode::mouseMoveEvent(event);
}

void BlackboardNode2::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        m_bDragging = false;
        updateBlackboard();
        emit nodePosChangedSignal();
        return;
    }
    if (isMoving()) {
        updateChildItemsPos();
    }
    ZenoNode::mouseReleaseEvent(event);
}

void BlackboardNode2::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    QCursor cursor;
    bool bDrag = isDragArea(event->pos());
    if (bDrag) {
        setCursor(QCursor(Qt::SizeFDiagCursor));
    } else {
        setCursor(QCursor(Qt::ArrowCursor));
    }
    ZenoNode::hoverMoveEvent(event);
}

bool BlackboardNode2::isDragArea(QPointF pos) {
    QPointF bottomright = boundingRect().bottomRight();
    QPointF offset = pos - bottomright;
    return (offset.manhattanLength() <  100);
}

void BlackboardNode2::updateBlackboard() {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    info.sz = this->size();
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), info, subGraphIndex(), true);
}

void BlackboardNode2::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    
    //draw background
    QFontMetrics fm(this->font());
    int margin = ZenoStyle::dpiScaled(8);
    int height = fm.height(); 
    QRectF rect = QRectF(0, height + margin, this->boundingRect().width(), this->boundingRect().height() - height - margin);
    QColor background = blackboard.background.isValid() ? blackboard.background : QColor(60, 70, 69);
    painter->setOpacity(0.3);
    painter->fillRect(rect, background);
    painter->setOpacity(1);
    QPen pen(background);
    pen.setWidthF(ZenoStyle::dpiScaled(2));
    painter->setPen(pen);
    painter->drawRect(rect);

    painter->setFont(this->font());
    painter->setPen("#FFFFFF");
    //draw title
    if (!blackboard.title.isEmpty()) {
        QRectF textRect(margin, 0, boundingRect().width() - 2 * margin, height);
        painter->drawText(textRect, Qt::AlignLeft, blackboard.title);
    }
    //draw content
    if (!blackboard.content.isEmpty()) {
        QRectF textRect(rect.x() + margin, rect.y() + margin, rect.width() - 2 * margin, rect.height() - 2 * margin);
        painter->drawText(textRect, Qt::AlignLeft | Qt::TextFlag::TextWordWrap, blackboard.content);
    }
    ZenoNode::paint(painter, option, widget);
}

QVariant BlackboardNode2::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == QGraphicsItem::ItemPositionHasChanged) {
        QPointF newPos = value.toPointF();
        QPointF oldPos = index().data(ROLE_OBJPOS).toPointF();
        for (auto item : m_childItems) {
            if (!item) {
                continue;
            }
            QPointF itemPos = item->index().data(ROLE_OBJPOS).toPointF();
            QPointF pos(itemPos.x() + (newPos.x() - oldPos.x()), itemPos.y() + (newPos.y() - oldPos.y()));
            item->setPos(pos);
        }
    }
    return ZenoNode::itemChange(change, value);
}
