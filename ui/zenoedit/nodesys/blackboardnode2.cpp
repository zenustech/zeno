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

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

GroupTextItem::GroupTextItem(QGraphicsItem *parent) : 
    QGraphicsWidget(parent), 
    m_bMoving(false)
{
    setFlags(ItemIsSelectable);    
}
GroupTextItem ::~GroupTextItem() {
}

void GroupTextItem::setText(const QString &text) {
    m_text = text;
    update();
}

void GroupTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event) 
{
    QGraphicsWidget::mousePressEvent(event);
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
    painter->setFont(font());
    QFontMetrics fontMetrics(font());
    QString text = m_text;
    if (fontMetrics.width(text) > boundingRect().width()) {
        text = fontMetrics.elidedText(text, Qt::ElideRight, boundingRect().width());
    }
    painter->drawText(QPointF(0, ZenoStyle::dpiScaled(10 / editor_factor)), text);
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
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), QVariant::fromValue(info), subGraphIndex(), false);
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
    } else if (m_childItems.contains(item)) {
        updateChildRelativePos(item);
        if (m_itemRelativeMap.contains(item->nodeId())) {
            QPointF pos = m_itemRelativeMap[item->nodeId()];
            item->updateNodePos(mapToScene(pos), false);
        }
    }
    return false;
}

void BlackboardNode2::onZoomed() 
{
    int fontSize = 12 / editor_factor > 12 ? 12 / editor_factor : 12;
    QFont font("HarmonyOS Sans", fontSize);
    QFontMetrics fontMetrics(font);
    m_pTextItem->resize(QSize(boundingRect().width(), fontMetrics.height()));
    m_pTextItem->setFont(font);
    QPointF pos = QPointF(0, -m_pTextItem->boundingRect().height());
    m_pTextItem->setPos(pos);
}

QRectF BlackboardNode2::boundingRect() const {
    return ZenoNode::boundingRect();
}

void BlackboardNode2::onUpdateParamsNotDesc() 
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    m_pTextItem->setText(blackboard.title);
}

void BlackboardNode2::appendChildItem(ZenoNode *item)
{
    m_childItems << item;
    item->setGroupNode(this);
    if (item->zValue() <= zValue()) {
        item->setZValue(zValue() + 1);
    }
    updateClidItem(true, item->nodeId());
    QPointF pos = mapFromItem(item, 0, 0);
    m_itemRelativeMap[item->nodeId()] = pos;
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
        item->updateNodePos(item->pos(), false);
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
        m_itemRelativeMap.remove(pNode->nodeId());
    }
}

void BlackboardNode2::updateChildRelativePos(const ZenoNode *item) 
{
    if (m_itemRelativeMap.contains(item->nodeId())) {
        m_itemRelativeMap[item->nodeId()] = mapFromItem(item, 0, 0);
    }
}

ZLayoutBackground *BlackboardNode2::initBodyWidget(ZenoSubGraphScene *pScene) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
        m_pTextItem->resize(QSize(boundingRect().width(), m_pTextItem->boundingRect().height()));
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
    }
    update();
    ZenoNode::mousePressEvent(event);
}

void BlackboardNode2::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        qreal ptop, pbottom, pleft, pright;
        QRectF rect = this->geometry();
        ptop = rect.top();
        pbottom = rect.bottom();
        pleft = rect.left();
        pright = rect.right();
        if (resizeDir & bottom) {
            if (rect.height() == minimumHeight()) {
                pbottom = max(event->scenePos().y(), ptop);
            }
            else if (rect.height() == maximumHeight()) {
                pbottom = min(event->scenePos().y(), ptop);
            }
            else {
                pbottom = event->scenePos().y();
            }
        
        }
        if (resizeDir & right) {
            if (rect.width() == minimumWidth()) {
                pright = max(event->scenePos().x(), pright);
            }
            else if (rect.width() == maximumWidth()) {
                pright = min(event->scenePos().x(), pright);
            }
            else {
                pright = event->scenePos().x();
            }
        }
        resize(pright - pleft, pbottom - ptop);
        m_pTextItem->resize(QSizeF(pright - pleft, m_pTextItem->boundingRect().height()));
        return;
    } 
    else
    {
        m_endPos = event->pos();
        QRectF selectArea = getSelectArea();
        QRectF rect = mapRectToScene(selectArea);
        for (auto item : m_childItems) {
            item->setSelected(rect.contains(item->pos()));
        }
        update();
        return;
    }
}

void BlackboardNode2::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        m_bDragging = false;
        updateBlackboard();
        emit nodePosChangedSignal();
        return;
    } 
    else
    {
        if (!m_endPos.isNull())
            setSelected(false);
        m_beginPos = QPointF();
        m_endPos = QPointF();
        update();
    }
}

void BlackboardNode2::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    isDragArea(event->pos());
    ZenoNode::hoverMoveEvent(event);
}

bool BlackboardNode2::isDragArea(QPointF pos) {
    QRectF rect = boundingRect();
    int diffLeft = pos.x() - rect.left(); 
    int diffRight = pos.x() - rect.right();
    int diffTop = pos.y() - rect.top() - 30;
    int diffBottom = pos.y() - rect.bottom();
    qreal width = 20;

    Qt::CursorShape cursorShape;
    if (abs(diffBottom) < width && diffBottom <= 0) {
        if (diffRight > -width && diffRight <= 0) {
            resizeDir = bottomRight;
            cursorShape = Qt::SizeFDiagCursor;

        }
        else {
            resizeDir = bottom;
            cursorShape = Qt::SizeVerCursor;
        }

    }    
    else if (abs(diffRight) < width) {
        resizeDir = right;
        cursorShape = Qt::SizeHorCursor;
    }
    else {
        resizeDir = nodir;
        cursorShape = Qt::ArrowCursor;
    }
    setCursor(cursorShape);
    return (resizeDir != nodir);
}

void BlackboardNode2::updateBlackboard() {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    info.sz = this->size();
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), QVariant::fromValue(info), subGraphIndex(), true);
}

void BlackboardNode2::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    
    //draw background
    QFontMetrics fm(this->font());
    int y = 0/*m_pTextItem->boundingRect().bottom()*/;
    QRectF rect = QRectF(0, y, this->boundingRect().width(), this->boundingRect().height() - y);
    QColor background = blackboard.background.isValid() ? blackboard.background : QColor(0, 100, 168);
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
        QRectF selectArea = getSelectArea();
        painter->fillRect(selectArea, background);
    }
    ZenoNode::paint(painter, option, widget);
}

QVariant BlackboardNode2::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == QGraphicsItem::ItemPositionHasChanged) {
        for (auto item : m_childItems) {
            if (!item || !m_itemRelativeMap.contains(item->nodeId())) {
                continue;
            }
            QPointF pos = mapToScene(m_itemRelativeMap[item->nodeId()]);
            item->setPos(pos);
        }
    }
    return ZenoNode::itemChange(change, value);
}

QRectF BlackboardNode2::getSelectArea()
{
    int x = (m_beginPos.x() < m_endPos.x()) ? m_beginPos.x() : m_endPos.x();
    int y = (m_beginPos.y() < m_endPos.y()) ? m_beginPos.y() : m_endPos.y();
    int w = qAbs(m_beginPos.x() - m_endPos.x()) + 1;
    int h = qAbs(m_beginPos.y() - m_endPos.y()) + 1;
    return QRectF(x, y, w, h);
}