#include "groupnode.h"
#include "uicommon.h"
#include "util/log.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "style/zenostyle.h"
#include <QPainter>
#include "model/graphsmanager.h"
#include "util/uihelper.h"
#include "nodeeditor/gv/zitemfactory.h"
#include "zenosubgraphscene.h"
#include <QtSvg/QSvgRenderer>
#include "variantptr.h"
#include "reflect/reflection.generated.hpp"


#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))

GroupTextItem::GroupTextItem(QGraphicsItem *parent) : 
    QGraphicsWidget(parent)
    , m_pLineEdit(nullptr)
{
    setFlags(ItemIsSelectable);
}
GroupTextItem ::~GroupTextItem() {
}

void GroupTextItem::setText(const QString &text) {
    m_text = text;
    update();
}

QString GroupTextItem::text() const
{
    return m_text;
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

void GroupTextItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    if (!m_pLineEdit)
    {
        m_pLineEdit = new ZEditableTextItem(m_text, this);
        m_pLineEdit->setPos(0, 0);
        connect(m_pLineEdit, &ZEditableTextItem::editingFinished, this, [=]() {
            QString text = m_pLineEdit->toPlainText();
            if (m_text != text)
            {
                m_text = m_pLineEdit->toPlainText();
                emit textChangedSignal(m_text);
            }
            m_pLineEdit->hide();
            update();
        });
    }
    else
    {
        m_pLineEdit->setText(m_text);
        m_pLineEdit->show();
    }
    m_pLineEdit->setFocus();
    update();
    QGraphicsWidget::mouseDoubleClickEvent(event);
}

void GroupTextItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if (m_pLineEdit && m_pLineEdit->isVisible())
    {
        if (m_pLineEdit->font() != font())
            m_pLineEdit->setFont(font());
        if (m_pLineEdit->boundingRect().size() != boundingRect().size())
            m_pLineEdit->setFixedSize(boundingRect().size());
        return;
    }
    painter->fillRect(boundingRect(), palette().color(QPalette::Window));
    QPen pen(palette().color(QPalette::Window));
    pen.setWidthF(ZenoStyle::scaleWidth(2));
    pen.setJoinStyle(Qt::MiterJoin);
    painter->setPen(pen);
    painter->drawRect(boundingRect());

    QColor color("#FFFFFF");
    painter->setPen(QPen(color));
    painter->setFont(font());
    QFontMetrics fontMetrics(font());
    QString text = m_text;
    qreal width = ZenoStyle::scaleWidth(4);
    QRectF textRect = boundingRect().adjusted(width, 0, -width, 0);
    if (fontMetrics.width(text) > textRect.width()) {
        text = fontMetrics.elidedText(text, Qt::ElideRight, textRect.width());
    }
    painter->drawText(textRect, Qt::AlignVCenter, text);
}

GroupNode::GroupNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(params, parent), 
    m_bDragging(false),
    m_bSelected(false),
    m_pTextItem(nullptr) 
{
    setAutoFillBackground(false);
    setAcceptHoverEvents(true);
    m_pTextItem = new GroupTextItem(this);    
    connect(m_pTextItem, &GroupTextItem::mouseMoveSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        _base::mouseMoveEvent(event);
    });
    connect(m_pTextItem, &GroupTextItem::mouseReleaseSignal, this, [=](QGraphicsSceneMouseEvent *event) { 
        _base::mouseReleaseEvent(event);
        m_bSelected = false;
    });
    connect(m_pTextItem, &GroupTextItem::mousePressSignal, this, [=](QGraphicsSceneMouseEvent *event) {
        _base::mousePressEvent(event);
        m_bSelected = true;
    });
    connect(m_pTextItem, &GroupTextItem::textChangedSignal, this, [=]() {
        updateBlackboard();
    });
    m_pTextItem->show();
    m_pTextItem->setZValue(0);
    onZoomed();
}

GroupNode::~GroupNode() {
}

void GroupNode::updateClidItem(bool isAdd, const QString nodeId)
{
    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS)))
    {
        int i = paramsM->indexFromName("items", true);
        if (i >= 0)
        {
            auto index = paramsM->index(i, 0);
            QString items = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).toString();
            QStringList itemList;
            if (!items.isEmpty())
                itemList = items.split(",");
            if (isAdd && !itemList.contains(nodeId)) {
                itemList << nodeId;
            }
            else if (!isAdd && itemList.contains(nodeId)) {
                itemList.removeOne(nodeId);
            }
            else {
                return;
            }
            UiHelper::qIndexSetData(index, QVariant::fromValue(UiHelper::qvarToAny(itemList.join(","), Param_String)), ROLE_PARAM_VALUE);
        }
    }
}

bool GroupNode::nodePosChanged(ZenoNodeBase*item)
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
            UiHelper::qIndexSetData(item->index(), mapToScene(pos), ROLE_OBJPOS);
        }
    }
    return false;
}

void GroupNode::onZoomed() 
{
    int fontSize = ZenoStyle::scaleWidth(12);
    QFont font = QApplication::font();
    font.setPointSize(fontSize);
    font.setBold(true);
    QFontMetrics fontMetrics(font);
    m_pTextItem->resize(QSizeF(boundingRect().width(), fontMetrics.height() + ZenoStyle::dpiScaled(10)));
    m_pTextItem->setFont(font);
    QPointF pos = QPointF(0, -m_pTextItem->boundingRect().height());
    m_pTextItem->setPos(pos);
}

QRectF GroupNode::boundingRect() const {
    QRectF rect = _base::boundingRect();
    rect.adjust(0, rect.y() - m_pTextItem->boundingRect().height(), 0, 0);
    return rect;
}

void GroupNode::appendChildItem(ZenoNodeBase*item)
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
        if (type == zeno::Node_Group) 
        {
            GroupNode *pNode = dynamic_cast<GroupNode *>(item);
            if (pNode) {
                pNode->updateChildItemsPos();
            }
        }
        UiHelper::qIndexSetData(item->index(), item->pos(), ROLE_OBJPOS);
    }
}

QVector<ZenoNodeBase*> GroupNode::getChildItems()
{
    return m_childItems;
}

void GroupNode::removeChildItem(ZenoNodeBase*pNode)
{
    if (m_childItems.contains(pNode)) {
        m_childItems.removeOne(pNode);
        pNode->setGroupNode(nullptr);
        m_itemRelativePosMap.remove(pNode->nodeId());
    }
}

void GroupNode::updateChildRelativePos(const ZenoNodeBase*item)
{
    if (m_itemRelativePosMap.contains(item->nodeId())) {
        m_itemRelativePosMap[item->nodeId()] = mapFromItem(item, 0, 0);
    }
}

void GroupNode::initLayout() {
    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS)))
    {
        auto index = paramsM->index(paramsM->indexFromName("title", true), 0);
        if (index.isValid())
        {
            QString title = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).toString();
            m_pTextItem->setText(title);
        }
        index = paramsM->index(paramsM->indexFromName("background", true), 0);
        if (index.isValid())
        {
            UI_VECTYPE background = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).value<UI_VECTYPE>();
            if (background.size() == 3)
            {
                QPalette palette = m_pTextItem->palette();
                QColor col = QColor::fromRgbF(background[0], background[1], background[2]);
                palette.setColor(QPalette::Window, col);
                m_pTextItem->setPalette(palette);
                setSvgData(col.name());
            }
        }
        index = paramsM->index(paramsM->indexFromName("size", true), 0);
        if (index.isValid())
        {
            UI_VECTYPE sizeVec = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).value<UI_VECTYPE>();
            if (sizeVec.size() == 2)
            {
                QSizeF size(sizeVec[0], sizeVec[1]);
                if (size != this->size()) {
                    resize(size);
                    emit nodePosChangedSignal();
                }
                if (size.width() != m_pTextItem->boundingRect().width())
                    m_pTextItem->resize(QSizeF(size.width(), m_pTextItem->boundingRect().height()));
            }
        }
    }
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    connect(paramsM, &ParamsModel::dataChanged, this, &GroupNode::onDataChanged);
}

void GroupNode::mousePressEvent(QGraphicsSceneMouseEvent *event) 
{
    QPointF pos = event->pos();
    if (isDragArea(pos)) {
        m_bDragging = true;
    } else {
        m_bDragging = false;
    }
    _base::mousePressEvent(event);
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
        QPointF oldPos = index().data(ROLE_OBJPOS).toPointF();
        if (oldPos == scenePos())
            emit nodePosChangedSignal(); //update childitems
        else
            UiHelper::qIndexSetData(this->index(), scenePos(), ROLE_OBJPOS);
        return;
    } 
}

void GroupNode::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    isDragArea(event->pos());
    _base::hoverMoveEvent(event);
}

bool GroupNode::isDragArea(QPointF pos) {
    QRectF rect = boundingRect();    
    rect.adjust(0, m_pTextItem->boundingRect().height(), 0, 0);
    int diffLeft = pos.x() - rect.left(); 
    int diffRight = pos.x() - rect.right();
    int diffTop = pos.y() - rect.top() ;
    int diffBottom = pos.y() - rect.bottom();
    qreal width = ZenoStyle::scaleWidth(16);

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
    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS)))
    {
        auto index = paramsM->index(paramsM->indexFromName("title", true), 0);
        auto strVal = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).toString();
        if (index.isValid() && strVal != m_pTextItem->text())
        {
            UiHelper::qIndexSetData(index, QVariant::fromValue(UiHelper::qvarToAny(m_pTextItem->text(), Param_String)), ROLE_PARAM_VALUE);
        }
        index = paramsM->index(paramsM->indexFromName("size", true), 0);
        if (index.isValid())
        {
            UI_VECTYPE val;
            val << this->size().width() << this->size().height();
            UiHelper::qIndexSetData(index, QVariant::fromValue(UiHelper::qvarToAny(QVariant::fromValue(val), Param_Vec3f)), ROLE_PARAM_VALUE);
        }
    }
}

void GroupNode::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) 
{
    _base::paint(painter, option, widget);

    QColor background(0, 100, 168);
    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(index().data(ROLE_PARAMS)))
    {
        auto index = paramsM->index(paramsM->indexFromName("background", true), 0);
        if (index.isValid())
        {
            UI_VECTYPE val = UiHelper::anyToQvar(index.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).value<UI_VECTYPE>();
            if (val.size() == 3)
                background = QColor::fromRgbF(val[0], val[1], val[2]);
        }
    }
    
    //draw background
    QFontMetrics fm(this->font());
    QRectF rect = QRectF(0, 0, this->boundingRect().width(), this->boundingRect().height() - m_pTextItem->boundingRect().height());
    painter->setOpacity(0.3);
    painter->fillRect(rect, background);
    painter->setOpacity(1);
    QPen pen(background);
    pen.setWidthF(ZenoStyle::scaleWidth(2));
    pen.setJoinStyle(Qt::MiterJoin);
    painter->setPen(pen);
    painter->drawRect(rect);
    qreal width = ZenoStyle::scaleWidth(16);
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
    return _base::itemChange(change, value);
}

void GroupNode::onDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (topLeft.data(ROLE_PARAM_NAME) == "title")
    {
        QString title = UiHelper::anyToQvar(topLeft.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).toString();
        m_pTextItem->setText(title);
    }
    else if (topLeft.data(ROLE_PARAM_NAME) == "background")
    {
        UI_VECTYPE background = UiHelper::anyToQvar(topLeft.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).value<UI_VECTYPE>();
        if (background.size() == 3)
        {
            QPalette palette = m_pTextItem->palette();
            QColor col = QColor::fromRgbF(background[0], background[1], background[2]);
            palette.setColor(QPalette::Window, col);
            m_pTextItem->setPalette(palette);
            setSvgData(col.name());
        }
    }
    else if (topLeft.data(ROLE_PARAM_NAME) == "size")
    {
        UI_VECTYPE sizeVec = UiHelper::anyToQvar(topLeft.data(ROLE_PARAM_VALUE).value<zeno::reflect::Any>()).value<UI_VECTYPE>();
        QSizeF size(sizeVec[0], sizeVec[1]);
        if (size != this->size()) {
            resize(size);
            emit nodePosChangedSignal();
        }
        if (size.width() != m_pTextItem->boundingRect().width())
            m_pTextItem->resize(QSizeF(size.width(), m_pTextItem->boundingRect().height()));
    }
}

void GroupNode::setSvgData(QString color) 
{
    QFile file(":/icons/nodeEditor_group_scale-adjustor.svg");
    file.open(QIODevice::ReadOnly);
    m_svgByte = file.readAll();
    m_svgByte.replace("#D8D8D8", color.toStdString().c_str());
}

void GroupNode::setSelected(bool selected)
{
    m_bSelected = true;
    _base::setSelected(selected);
    m_bSelected = false;
}