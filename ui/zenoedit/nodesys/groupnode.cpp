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
void GroupTextItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
        emit mouseReleaseSignal(event);
}

void GroupTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QColor color = this->palette().color(QPalette::WindowText);
    painter->setPen(QPen(color));    
    painter->setFont(font());
    QFontMetrics fontMetrics(font());
    QString text = m_text;
    if (fontMetrics.width(text) > boundingRect().width()) {
        text = fontMetrics.elidedText(text, Qt::ElideRight, boundingRect().width());
    }
    painter->drawText(this->boundingRect(), Qt::AlignVCenter, text);
}

GroupNode::GroupNode(const NodeUtilParam &params, QGraphicsItem *parent)
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
    connect(m_pTextItem, &GroupTextItem::mouseMoveSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        ZenoNode::mouseMoveEvent(event);
    });
    connect(m_pTextItem, &GroupTextItem::mouseReleaseSignal, this, [=](QGraphicsSceneMouseEvent *event) { 
        ZenoNode::mouseReleaseEvent(event);
    });
    connect(m_pTextItem, &GroupTextItem::mousePressSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        ZenoNode::mousePressEvent(event);
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
    QFont font("Alibaba PuHuiTi", fontSize);
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
    palette.setColor(QPalette::WindowText, blackboard.background);
    m_pTextItem->setPalette(palette);
    setSvgData(blackboard.background.name());
    if (blackboard.sz.isValid() && blackboard.sz != this->size()) {
        resize(blackboard.sz);
        emit nodePosChangedSignal();
    }
}

void GroupNode::appendChildItem(ZenoNode *item)
{
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
    palette.setColor(QPalette::WindowText, blackboard.background);
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
    if (!m_bDragging)
    {
        m_beginPos = event->pos();
    }
    update();
    ZenoNode::mousePressEvent(event);
}

void GroupNode::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
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
            QRectF itemRect(item->scenePos(), item->boundingRect().size());
            QRectF interRect = rect.intersected(itemRect);
            item->setSelected(interRect.isValid());
        }
        update();
        return;
    }
}

void GroupNode::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
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

void GroupNode::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    isDragArea(event->pos());
    ZenoNode::hoverMoveEvent(event);
}

bool GroupNode::isDragArea(QPointF pos) {
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
    qreal width = ZenoStyle::dpiScaled(16);
    QSvgRenderer svgRender(m_svgByte);
    svgRender.render(painter, QRectF(boundingRect().bottomRight() - QPointF(width, width), boundingRect().bottomRight()));
}

QVariant GroupNode::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == QGraphicsItem::ItemPositionHasChanged) {
        for (auto item : m_childItems) {
            if (!item || !m_itemRelativePosMap.contains(item->nodeId())) {
                continue;
            }
            QPointF pos = mapToScene(m_itemRelativePosMap[item->nodeId()]);
            item->setPos(pos);
        }
    }
    return ZenoNode::itemChange(change, value);
}

QRectF GroupNode::getSelectArea()
{
    int x = (m_beginPos.x() < m_endPos.x()) ? m_beginPos.x() : m_endPos.x();
    int y = (m_beginPos.y() < m_endPos.y()) ? m_beginPos.y() : m_endPos.y();
    int w = qAbs(m_beginPos.x() - m_endPos.x()) + 1;
    int h = qAbs(m_beginPos.y() - m_endPos.y()) + 1;
    return QRectF(x, y, w, h);
}

void GroupNode::setSvgData(QString color) 
{
    QFile file(":/icons/nodeEditor_group_scale-adjustor.svg");
    file.open(QIODevice::ReadOnly);
    m_svgByte = file.readAll();
    m_svgByte.replace("#D8D8D8", color.toStdString().c_str());
}
