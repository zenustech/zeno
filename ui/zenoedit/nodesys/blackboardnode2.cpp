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
#include "zenosubgraphscene.h"

GroupTextItem::GroupTextItem(QGraphicsItem *parent) : 
    QGraphicsWidget(parent), 
    m_bMoving(false)
{
    setFlags(ItemIsSelectable);
    setFlag(QGraphicsItem::ItemIgnoresTransformations);
}
GroupTextItem ::~GroupTextItem() {
}

void GroupTextItem::setText(const QString &text) {
    m_text = text;
    update();
}

void GroupTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event) 
{
    emit mousePressSignal();
    m_beginPos = event->pos();
}
void GroupTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event) 
{
    m_bMoving = true;
    emit posChangedSignal(event->pos() - m_beginPos);
}
void GroupTextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bMoving) {
        emit updatePosSignal();
        m_bMoving = false;
    }
}

void GroupTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    painter->setPen(QPen("#FFF"));
    QFont font("HarmonyOS Sans", 12);
    painter->setFont(font);
    QFontMetrics fontMetrics(font);
    QString text = m_text;
    if (fontMetrics.width(text) > boundingRect().width()) {
        text = fontMetrics.elidedText(text, Qt::ElideRight, boundingRect().width());
    }
    painter->drawText(QPointF(0, boundingRect().bottom() - ZenoStyle::dpiScaled(10)), text);
}

BlackboardNode2::BlackboardNode2(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent), 
    m_bDragging(false),
    m_bSelecting(false),
    m_beginPos(QPointF()),
    m_endPos(QPointF()), 
    m_pTextItem(nullptr) 
{
    setAutoFillBackground(false);
    setAcceptHoverEvents(true);
    m_pTextItem = new GroupTextItem(this);    
    connect(m_pTextItem, &GroupTextItem::posChangedSignal, this, [=](const QPointF &pos) {
        QPointF newPos = scenePos() + pos;
        setPos(newPos);
    });
    connect(m_pTextItem, &GroupTextItem::updatePosSignal, this, [=]() { 
        updateNodePos(scenePos());
        updateChildItemsPos();
    });
    connect(m_pTextItem, &GroupTextItem::mousePressSignal, this, [=]() {
        setSelected(true);
    });
    m_pTextItem->show();
    m_pTextItem->setZValue(0);
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
        appendChildItem(item);
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
        removeChildItem(item);
        updateClidItem(false, item->nodeId());
    }
    return false;
}

void BlackboardNode2::onZoomed() 
{
    m_pTextItem->resize(QSize(boundingRect().width() * editor_factor, ZenoStyle::dpiScaled(32)));
    QPointF pos = QPointF(0, -m_pTextItem->boundingRect().height() / editor_factor);
    m_pTextItem->setPos(pos);
}

QRectF BlackboardNode2::boundingRect() const {
    return ZenoNode::boundingRect();
}

void BlackboardNode2::onUpdateParamsNotDesc() 
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    //remove items
    //m_pTextItem->setText(blackboard.title);
    //for (auto item : m_childItems) {
    //    if (!blackboard.items.contains(item->nodeId())) {
    //        removeChildItem(item);
    //    }
    //}
    ////add Items
    //for (auto id : blackboard.items) {
    //    bool isExist = false;
    //    for (auto item : m_childItems) {
    //        if (item->nodeId() == id) {
    //            isExist = true;
    //            break;
    //        }
    //    }
    //    if (!isExist) {
    //        ZenoSubGraphScene *pScene = qobject_cast<ZenoSubGraphScene *>(this->scene());
    //        if (pScene) {
    //            QGraphicsItem *pItem = pScene->getNode(id);
    //            ZenoNode *pNode = dynamic_cast<ZenoNode *>(pItem);
    //            if (pNode)
    //                appendChildItem(pNode);
    //        }
    //    }
    //}
}

void BlackboardNode2::appendChildItem(ZenoNode *item)
{
    m_childItems << item;
    item->setGroupNode(this);
    if (item->zValue() <= zValue()) {
        item->setZValue(zValue() + 1);
    }
    updateClidItem(true, item->nodeId());
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
        pNode->setGroupNode(nullptr);
    }
}

ZLayoutBackground *BlackboardNode2::initBodyWidget(ZenoSubGraphScene *pScene) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
        m_pTextItem->resize(QSize(boundingRect().width() * editor_factor, ZenoStyle::dpiScaled(32)));
    }
    m_pTextItem->setText(blackboard.title);
    return new ZLayoutBackground(this);
}

ZLayoutBackground *BlackboardNode2::initHeaderWidget(IGraphsModel*)
{
    return new ZLayoutBackground(this);    
}

void BlackboardNode2::mousePressEvent(QGraphicsSceneMouseEvent *event) 
{
    QPointF pos = event->pos();
    if (isDragArea(pos)) {
        m_bDragging = true;
    } else {
        m_bDragging = false;
    }
    if (!m_bDragging)
    {
        m_beginPos = event->pos();
        setSelected(false);
    }
    update();
}

void BlackboardNode2::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        QPointF topLeft = sceneBoundingRect().topLeft();
        QPointF newPos = event->scenePos();
        QPointF currPos = sceneBoundingRect().bottomRight();

        qreal newWidth = newPos.x() - topLeft.x();
        qreal newHeight = newPos.y() - topLeft.y();
        resize(newWidth, newHeight);
        m_pTextItem->resize(QSizeF(newWidth * editor_factor, m_pTextItem->boundingRect().height()));
        return;
    } 
    else
    {
        m_endPos = event->pos();
        update();
        return;
    }

    //ZenoNode::mouseMoveEvent(event);
}

void BlackboardNode2::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        m_bDragging = false;
        updateBlackboard();
        emit nodePosChangedSignal();
        return;
    }
    m_beginPos = QPointF();
    m_endPos = QPointF();
    update();
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
    int y = 0/*m_pTextItem->boundingRect().bottom()*/;
    QRectF rect = QRectF(0, y, this->boundingRect().width(), this->boundingRect().height() - y);
    QColor background = blackboard.background.isValid() ? blackboard.background : QColor(60, 70, 69);
    if (!index().data(ROLE_COLLASPED).toBool())
    {
        painter->setOpacity(0.3);
        painter->fillRect(rect, background);
        painter->setOpacity(1);
        QPen pen(background);
        pen.setWidthF(ZenoStyle::dpiScaled(2));
        painter->setPen(pen);
        painter->drawRect(rect);
    }

    if (!m_beginPos.isNull() && !m_endPos.isNull() && m_beginPos != m_endPos) 
    {
        painter->setOpacity(0.5);
        int x = (m_beginPos.x() < m_endPos.x()) ? m_beginPos.x() : m_endPos.x();
        int y = (m_beginPos.y() < m_endPos.y()) ? m_beginPos.y() : m_endPos.y();
        int w = qAbs(m_beginPos.x() - m_endPos.x()) + 1;
        int h = qAbs(m_beginPos.y() - m_endPos.y()) + 1;
        QRectF selectArea(x, y, w, h);
        painter->fillRect(selectArea, background);
        QRectF rect = mapRectToScene(selectArea);
        for (auto item : m_childItems) {
            item->setSelected(rect.contains(item->pos()));
        }
    }
    QPointF p = index().data(ROLE_OBJPOS).toPointF();
    QPointF p1 = m_pTextItem->pos();
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

 void BlackboardNode2::onCollaspeUpdated(bool collasped) {
    for (auto item : m_childItems) {
        item->setVisible(!collasped);
    }
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    QFontMetrics fm(this->font());
    int height = fm.height() + ZenoStyle::dpiScaled(8); 
    resize(blackboard.sz.width(), collasped ? height : blackboard.sz.height());
    update();
}