#include "groupnode.h"
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
#include <QtSvg/QSvgRenderer>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

GroupTextItem::GroupTextItem(QGraphicsItem *parent) : 
    QGraphicsWidget(parent)
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
    emit mousePressSignal(event);
}
void GroupTextItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event) 
{
    emit mouseMoveSignal(event);
}
void GroupTextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) 
{
    emit mouseReleaseSignal(event);
}

void GroupTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    qreal width = ZenoStyle::dpiScaled(1);
    painter->fillRect(boundingRect().adjusted(-width, -width, width, 0), palette().color(QPalette::Window));

    QColor color("#FFFFFF");
    painter->setPen(QPen(color));
    painter->setFont(font());
    QFontMetrics fontMetrics(font());
    QString text = m_text;
    width = ZenoStyle::dpiScaled(4);
    QRectF textRect = boundingRect().adjusted(width, 0, -width, 0);
    if (fontMetrics.width(text) > textRect.width()) {
        text = fontMetrics.elidedText(text, Qt::ElideRight, textRect.width());
    }
    painter->drawText(textRect, Qt::AlignVCenter, text);
}

GroupNode::GroupNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent), 
    m_bDragging(false),
    m_bSelected(false),
    m_pTextItem(nullptr) 
{
    setAutoFillBackground(false);
    setAcceptHoverEvents(true);
    m_pTextItem = new GroupTextItem(this);    
    connect(m_pTextItem, &GroupTextItem::mouseMoveSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        ZenoNode::mouseMoveEvent(event);
    });
    connect(m_pTextItem, &GroupTextItem::mouseReleaseSignal, this, [=](QGraphicsSceneMouseEvent *event) { 
        ZenoNode::mouseReleaseEvent(event);
        m_bSelected = false;
    });
    connect(m_pTextItem, &GroupTextItem::mousePressSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        ZenoNode::mousePressEvent(event);
        m_bSelected = true;
    });
    m_pTextItem->show();
    m_pTextItem->setZValue(0);
    onZoomed();
}

GroupNode::~GroupNode() {
}

void GroupNode::updateClidItem(bool isAdd, const QString nodeId)
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

bool GroupNode::nodePosChanged(ZenoNode *item) 
{
    if (this->sceneBoundingRect().contains(item->sceneBoundingRect()) && !m_childItems.contains(item)) {

        GroupNode *pParentItem = item->getGroupNode();
        if (getGroupNode() == item)
            return false;
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
        GroupNode *pGroupNode = this->getGroupNode();
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
        if (m_itemRelativePosMap.contains(item->nodeId())) {
            QPointF pos = m_itemRelativePosMap[item->nodeId()];
            item->updateNodePos(mapToScene(pos), false);
        }
    }
    return false;
}

void GroupNode::onZoomed() 
{
    int fontSize = 12 / editor_factor > 12 ? 12 / editor_factor : 12;
    QFont font = zenoApp->font();
    font.setPointSize(fontSize);
    font.setBold(true);
    QFontMetrics fontMetrics(font);
    m_pTextItem->resize(QSize(boundingRect().width(), fontMetrics.height() + ZenoStyle::dpiScaled(10)));
    m_pTextItem->setFont(font);
    QPointF pos = QPointF(0, -m_pTextItem->boundingRect().height());
    m_pTextItem->setPos(pos);
}

QRectF GroupNode::boundingRect() const {
    QRectF rect = ZenoNode::boundingRect();
    rect.adjust(0, rect.y() - m_pTextItem->boundingRect().height(), 0, 0);
    return rect;
}

void GroupNode::onUpdateParamsNotDesc() 
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    m_pTextItem->setText(blackboard.title);
    QPalette palette = m_pTextItem->palette();
    palette.setColor(QPalette::Window, blackboard.background);
    m_pTextItem->setPalette(palette);
    setSvgData(blackboard.background.name());
    if (blackboard.sz.isValid() && blackboard.sz != this->size()) {
        resize(blackboard.sz);
        emit nodePosChangedSignal();
    }
    if (blackboard.sz.width() != m_pTextItem->boundingRect().width())
        m_pTextItem->resize(QSize(blackboard.sz.width(), m_pTextItem->boundingRect().height()));
}

void GroupNode::appendChildItem(ZenoNode *item)
{
    if (item->getGroupNode()) 
    {
        item->getGroupNode()->removeChildItem(item);
    }
    m_childItems << item;
    item->setGroupNode(this);
    if (item->zValue() <= zValue()) {
        item->setZValue(zValue() + 1);
    }
    updateClidItem(true, item->nodeId());
    QPointF pos = mapFromItem(item, 0, 0);
    m_itemRelativePosMap[item->nodeId()] = pos;
}

void GroupNode::updateChildItemsPos()
{
    for (auto item : m_childItems) {
        if (!item) {
            continue;
        }
        int type = item->index().data(ROLE_NODETYPE).toInt();
        if (type == GROUP_NODE) 
        {
            GroupNode *pNode = dynamic_cast<GroupNode *>(item);
            if (pNode) {
                pNode->updateChildItemsPos();
            }
        }
        item->updateNodePos(item->pos(), false);
    }
}

QVector<ZenoNode *> GroupNode::getChildItems() 
{
    return m_childItems;
}

void GroupNode::removeChildItem(ZenoNode *pNode) 
{
    if (m_childItems.contains(pNode)) {
        m_childItems.removeOne(pNode);
        pNode->setGroupNode(nullptr);
        m_itemRelativePosMap.remove(pNode->nodeId());
    }
}

void GroupNode::updateChildRelativePos(const ZenoNode *item) 
{
    if (m_itemRelativePosMap.contains(item->nodeId())) {
        m_itemRelativePosMap[item->nodeId()] = mapFromItem(item, 0, 0);
    }
}

ZLayoutBackground *GroupNode::initBodyWidget(ZenoSubGraphScene *pScene) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
        m_pTextItem->resize(QSize(boundingRect().width(), m_pTextItem->boundingRect().height()));
    }
    QPalette palette = m_pTextItem->palette();
    palette.setColor(QPalette::Window, blackboard.background);
    m_pTextItem->setPalette(palette);
    m_pTextItem->setText(blackboard.title);
    setSvgData(blackboard.background.name());
    return new ZLayoutBackground(this);
}

ZLayoutBackground *GroupNode::initHeaderWidget(IGraphsModel*)
{
    return new ZLayoutBackground(this);    
}

void GroupNode::mousePressEvent(QGraphicsSceneMouseEvent *event) 
{
    QPointF pos = event->pos();
    if (isDragArea(pos)) {
        m_bDragging = true;
    } else {
        m_bDragging = false;
    }
    ZenoNode::mousePressEvent(event);
}

void GroupNode::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        qreal ptop, pbottom, pleft, pright;
        ptop = scenePos().y();
        pbottom = scenePos().y() + size().height();
        pleft = scenePos().x();
        pright = scenePos().x() + size().width();
        if (resizeDir & top) {
            if (size().height() == minimumHeight()) {
                ptop = min(event->scenePos().y(), ptop);
            }
            else if (size().height() == maximumHeight()) {
                ptop = max(event->scenePos().y(), ptop);
            }
            else {
                ptop = event->scenePos().y();
            }
        }
        else if (resizeDir & bottom) {
            if (size().height() == minimumHeight()) {
                pbottom = max(event->scenePos().y(), ptop);
            }
            else if (size().height() == maximumHeight()) {
                pbottom = min(event->scenePos().y(), ptop);
            }
            else {
                pbottom = event->scenePos().y();
            }
        
        }
        if (resizeDir & left) {
            if (size().width() == minimumWidth()) {
                pleft = min(event->scenePos().x(), pleft);
            }
            else if (size().width() == maximumWidth()) {
                pleft = max(event->scenePos().x(), pleft);
            }
            else {
                pleft = event->scenePos().x();
            }
        }
        else if (resizeDir & right) {
            if (size().width() == minimumWidth()) {
                pright = max(event->scenePos().x(), pright);
            }
            else if (size().width() == maximumWidth()) {
                pright = min(event->scenePos().x(), pright);
            }
            else {
                pright = event->scenePos().x();
            }
        }

        resize(pright - pleft, pbottom - ptop);
        if ((resizeDir & left) || (resizeDir & top)) {
            setPos(pleft, ptop);
            for (auto item : m_childItems) {
                updateChildRelativePos(item);
            }
        }
        m_pTextItem->resize(QSizeF(pright - pleft, m_pTextItem->boundingRect().height()));
        return;
    } 
}

void GroupNode::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (m_bDragging) {
        m_bDragging = false;
        updateBlackboard();
        updateNodePos(scenePos());
        return;
    } 
}

void GroupNode::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    isDragArea(event->pos());
    ZenoNode::hoverMoveEvent(event);
}

bool GroupNode::isDragArea(QPointF pos) {
    QRectF rect = boundingRect();    
    rect.adjust(0, m_pTextItem->boundingRect().height(), 0, 0);
    int diffLeft = pos.x() - rect.left(); 
    int diffRight = pos.x() - rect.right();
    int diffTop = pos.y() - rect.top() ;
    int diffBottom = pos.y() - rect.bottom();
    qreal width = 50;

    Qt::CursorShape cursorShape(Qt::ArrowCursor);
    resizeDir = nodir;
    if (rect.contains(pos)) {
        if (diffTop < width && diffTop >= 0) {
            if (diffLeft < width && diffLeft >= 0) {
                resizeDir = topLeft;
                cursorShape = Qt::SizeFDiagCursor;
            } else if (diffRight > -width && diffRight <= 0) {
                resizeDir = topRight;
                cursorShape = Qt::SizeBDiagCursor;
            } else {
                resizeDir = top;
                cursorShape = Qt::SizeVerCursor;
            }
        } else if (abs(diffBottom) < width && diffBottom <= 0) {
            if (diffLeft < width && diffLeft >= 0) {
                resizeDir = bottomLeft;
                cursorShape = Qt::SizeBDiagCursor;
            } else if (diffRight > -width && diffRight <= 0) {
                resizeDir = bottomRight;
                cursorShape = Qt::SizeFDiagCursor;

            } else {
                resizeDir = bottom;
                cursorShape = Qt::SizeVerCursor;
            }
        } else if (abs(diffLeft) < width) {
            resizeDir = left;
            cursorShape = Qt::SizeHorCursor;
        } else if (abs(diffRight) < width) {
            resizeDir = right;
            cursorShape = Qt::SizeHorCursor;
        }
    }
    setCursor(cursorShape);
    bool result = resizeDir != nodir;
    setAcceptedMouseButtons(result ? Qt::LeftButton : Qt::NoButton);
    return result;
}

void GroupNode::updateBlackboard() {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    info.sz = this->size();
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), QVariant::fromValue(info), subGraphIndex(), true);
}

void GroupNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    ZenoNode::paint(painter, option, widget);
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    
    //draw background
    QFontMetrics fm(this->font());
    QRectF rect = QRectF(0, 0, this->boundingRect().width(), this->boundingRect().height() - m_pTextItem->boundingRect().height());
    QColor background = blackboard.background.isValid() ? blackboard.background : QColor(0, 100, 168);
    painter->setOpacity(0.3);
    painter->fillRect(rect, background);
    painter->setOpacity(1);
    QPen pen(background);
    pen.setWidthF(ZenoStyle::dpiScaled(2));
    pen.setJoinStyle(Qt::MiterJoin);
    painter->setPen(pen);
    painter->drawRect(rect);
    qreal width = ZenoStyle::dpiScaled(16);
    QSvgRenderer svgRender(m_svgByte);
    svgRender.render(painter, QRectF(boundingRect().bottomRight() - QPointF(width, width), boundingRect().bottomRight()));
}

QVariant GroupNode::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == QGraphicsItem::ItemPositionHasChanged && !m_bDragging) {
        for (auto item : m_childItems) {
            if (!item || !m_itemRelativePosMap.contains(item->nodeId())) {
                continue;
            }
            QPointF pos = mapToScene(m_itemRelativePosMap[item->nodeId()]);
            item->setPos(pos);
        }
    } 
    else if (change == QGraphicsItem::ItemPositionChange) 
    {
        setMoving(true);
        return value;
    }
    else if(change == QGraphicsItem::ItemSelectedHasChanged) 
    {
        QPainterPath path = scene()->selectionArea();
        if (isSelected() && !m_bSelected && !path.contains(sceneBoundingRect()))
            this->setSelected(false);
    }
    return ZenoNode::itemChange(change, value);
}

void GroupNode::setSvgData(QString color) 
{
    QFile file(":/icons/nodeEditor_group_scale-adjustor.svg");
    file.open(QIODevice::ReadOnly);
    m_svgByte = file.readAll();
    m_svgByte.replace("#D8D8D8", color.toStdString().c_str());
}