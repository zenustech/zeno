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
    m_bDragging(false), 
    m_pTitle(nullptr), 
    m_pTextEdit(nullptr), 
    m_mainLayout(nullptr), 
    m_pMainSpaceItem(nullptr) 
{
    setAutoFillBackground(false);
    setAcceptHoverEvents(true);
    QFont font("HarmonyOS Sans", 12);
    this->setFont(font);
    initUI();
}

BlackboardNode2::~BlackboardNode2() {
}

void BlackboardNode2::initUI() {
    //init title
    LineEditParam param;
    param.palette = this->palette();
    param.propertyParam = "blackboard_title";
    param.font = this->font();
    m_pTitle = new ZenoParamLineEdit("", CONTROL_STRING, param);

    connect(m_pTitle, &ZenoParamLineEdit::editingFinished, this, [=]() {
        PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
        if (info.title != m_pTitle->text()) {
            updateBlackboard();
        }
    });

    QGraphicsLinearLayout *pTitlelayout = new QGraphicsLinearLayout(Qt::Horizontal);
    m_pTitle->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    pTitlelayout->addItem(m_pTitle);
    ZenoSpacerItem *pSpaceItem = new ZenoSpacerItem(true, 100);
    pSpaceItem->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    pTitlelayout->addItem(pSpaceItem);
    pTitlelayout->setContentsMargins(0, 0, 0, 0);

    //init content
    m_pTextEdit = new ZenoParamBlackboard("", param);
    m_pTextEdit->hide();
    if (m_pTextEdit == nullptr)
        return;
    m_pTextEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    connect(m_pTextEdit, &ZenoParamBlackboard::editingFinished, this, [=]() {
        updateView(false);
        PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
        if (info.content != m_pTextEdit->text()) {
            updateBlackboard();
        }
    });

    m_pMainSpaceItem = new ZenoSpacerItem(false, this->boundingRect().height()-m_pTitle->boundingRect().height());
    m_pMainSpaceItem->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_mainLayout = new QGraphicsLinearLayout(Qt::Vertical, this);
    m_mainLayout->setSpacing(0);
    m_mainLayout->setContentsMargins(0, 0, 0, 0);
    m_mainLayout->addItem(pTitlelayout);
    m_mainLayout->addItem(m_pMainSpaceItem);    
    resize(500, 500);
}

void BlackboardNode2::updateView(bool isEditing) {
    if (m_mainLayout->count() > 1)
    {
        m_mainLayout->removeAt(1);
    }
    if (isEditing) {
        m_mainLayout->addItem(m_pTextEdit);
        m_pTextEdit->show();
        m_pTextEdit->foucusInEdit();
    }
    else {
        m_mainLayout->addItem(m_pMainSpaceItem);
        m_pTextEdit->hide();
    }
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

bool BlackboardNode2::nodePosChanged(ZenoNode *item) {
    if (this->sceneBoundingRect().contains(item->sceneBoundingRect()) && item->parentItem() != this) {
        if (item->parentItem() && item->parentItem()->sceneBoundingRect().contains(item->sceneBoundingRect()) &&
            (!item->parentItem()->sceneBoundingRect().contains(this->sceneBoundingRect()))) {
            return false;
        }
        QPointF pos = mapFromItem(item, 0, 0);
        item->setPos(pos);
        item->setParentItem(this);
        item->setMoving(false);
        item->setZValue(1);
        update();
        updateClidItem(true, item->nodeId());
        item->updateNodePos(pos);
        return true;
    } else if ((!this->sceneBoundingRect().contains(item->sceneBoundingRect())) && item->parentItem() == this) {
        QGraphicsItem *newParent = this->parentItem();
        while (dynamic_cast<BlackboardNode2*>(newParent)) {
            BlackboardNode2 *pBlackboardNode = dynamic_cast<BlackboardNode2 *>(newParent);
            bool isUpdate = pBlackboardNode->nodePosChanged(item);
            if (!isUpdate)
                newParent = pBlackboardNode->parentItem();
            else
                return true;
        }
        QPointF pos = item->mapToItem(newParent, 0, 0);
        item->setParentItem(newParent);
        item->setPos(pos);
        update();
        updateClidItem(false, item->nodeId());
        item->updateNodePos(pos);
    }
    return false;
}

void BlackboardNode2::updateFontSize(qreal factor) 
{
    int fontSize = 12 / factor > 12 ? 12 / factor : 12;
    QFont font("HarmonyOS Sans", fontSize);
    this->setFont(font);
    m_pTitle->setFont(font);
    m_pTextEdit->updateStyleSheet(fontSize);
}

QRectF BlackboardNode2::boundingRect() const {
    return ZenoNode::boundingRect();
}

void BlackboardNode2::onUpdateParamsNotDesc() 
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    m_pTextEdit->setText(info.content);
    m_pTitle->setText(info.title);
    if (info.sz.isValid()) {
        resize(info.sz);
    }
    QPalette palette = this->palette();
    palette.setColor(QPalette::Window, info.background);
    setPalette(palette);
}

ZLayoutBackground *BlackboardNode2::initBodyWidget(ZenoSubGraphScene *pScene) {
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();
    m_pTitle->setText(blackboard.title);
    m_pTextEdit->setText(blackboard.content);
    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
    }
    QPalette palette;
    palette.setColor(QPalette::Window, blackboard.background);
    palette.setColor(QPalette::WindowText, QColor("#FFFFFF"));
    setPalette(palette);
    return new ZLayoutBackground(this);
}

ZLayoutBackground *BlackboardNode2::initHeaderWidget() {
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
        emit nodePosChangedSignal();
        return;
    }

    QGraphicsWidget::mouseMoveEvent(event);
}

void BlackboardNode2::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (!isMoving() && event->pos().y() > m_pTitle->boundingRect().height() && !m_bDragging) {
        updateView(true);
    }
    if (m_bDragging) {
        m_bDragging = false;
        updateBlackboard();
        return;
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
    if (m_pTitle) {
        info.title = m_pTitle->text();
    }
    if (m_pTextEdit) {
        info.content = m_pTextEdit->text();
    }
    info.sz = this->size();
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), info, subGraphIndex(), true);
}

void BlackboardNode2::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    painter->setOpacity(0.3);
    int margin = ZenoStyle::dpiScaled(8);
    int height = m_pTitle->boundingRect().height(); 
    QRectF rect = QRectF(0, height + margin, this->boundingRect().width(), this->boundingRect().height() - height - margin);
    painter->fillRect(rect, palette().window().color());
    painter->setOpacity(1);
    QPen pen(palette().window().color());
    pen.setWidthF(ZenoStyle::dpiScaled(2));
    painter->setPen(pen);    
    painter->drawRect(rect);
    if (!m_pTextEdit->text().isEmpty() && !m_pTextEdit->isVisible()) {
        QRectF textRect(rect.x() + margin, rect.y() + margin, rect.width() - 2 * margin, rect.height() - 2 * margin);        
        painter->setFont(this->font());
        painter->setPen(palette().windowText().color());        
        painter->drawText(textRect, Qt::AlignLeft | Qt::TextFlag::TextWordWrap, m_pTextEdit->text());
    }
    ZenoNode::paint(painter, option, widget);
}
