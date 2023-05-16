#include "blackboardnode.h"
#include <zenoui/render/common_id.h>
#include "zenoapplication.h"
#include <zenomodel/include/graphsmanagment.h>
#include "util/log.h"
#include <zenoui/style/zenostyle.h>


BlackboardNode::BlackboardNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : ZenoNode(params, parent)
    , m_bDragging(false)
    , m_pTitle(nullptr)
    , m_pContent(nullptr)
{
}

BlackboardNode::~BlackboardNode()
{
}

QRectF BlackboardNode::boundingRect() const
{
    return ZenoNode::boundingRect();
}

void BlackboardNode::onUpdateParamsNotDesc()
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    m_pContent->setText(info.content);
    m_pTitle->setPlainText(info.title);
    if (info.sz.isValid())
    {
        resize(info.sz);
    }
}

ZLayoutBackground* BlackboardNode::initHeaderWidget(IGraphsModel*)
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground(this);
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
    headerWidget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);
    headerWidget->setBorder(headerBg.border_witdh, headerBg.clr_border);

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);

    ZenoSpacerItem *pSpacerItem = new ZenoSpacerItem(true, 100);

    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();

    m_pTitle = new ZGraphicsTextItem(blackboard.title, m_renderParams.nameFont, m_renderParams.nameClr.color(), this);
    m_pTitle->setText(blackboard.title);
    m_pTitle->setTextInteractionFlags(Qt::TextEditorInteraction);
    connect(m_pTitle->document(), &QTextDocument::contentsChanged, this, [=]() {
        ZGraphicsLayout::updateHierarchy(m_pTitle);
    });
    connect(m_pTitle, &ZGraphicsTextItem::editingFinished, this, [=]() {
        PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
        if (info.title != m_pTitle->toPlainText()) {
            updateBlackboard();
        }
    });

    ZGraphicsLayout* pNameLayout = new ZGraphicsLayout(Qt::Horizontal);
    pNameLayout->addItem(m_pTitle);
    pNameLayout->setContentsMargin(5, 5, 5, 5);

    int options = index().data(ROLE_OPTIONS).toInt();

    pHLayout->addLayout(pNameLayout);
    pHLayout->addItem(pSpacerItem);
    pHLayout->setSpacing(0);
    pHLayout->setContentsMargin(0, 0, 0, 0);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    headerWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    QColor clr(98, 108, 111);
    headerWidget->setColors(false, clr, clr, clr);

    return headerWidget;
}

ZLayoutBackground* BlackboardNode::initBodyWidget(ZenoSubGraphScene* pScene)
{
    ZLayoutBackground* bodyWidget = new ZLayoutBackground(this);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(false, QColor("#000000"), QColor("#000000"), QColor("#000000"));
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(true);
    qreal border = m_renderParams.bodyBg.border_witdh;
    pVLayout->setContentsMargin(border, border, border, border);

    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO blackboard = params["blackboard"].value.value<BLACKBOARD_INFO>();

    m_pContent = new ZGraphicsTextItem(blackboard.content, m_renderParams.nameFont, m_renderParams.nameClr.color(), this);
    m_pContent->setTextInteractionFlags(Qt::TextEditorInteraction);

    int h = m_pTitle->boundingRect().height();
    QSize sz(blackboard.sz.width(), blackboard.sz.height() - h);
    m_pContent->setData(GVKEY_SIZEHINT, sz);

    pVLayout->addItem(m_pContent);

    connect(m_pContent, &ZGraphicsTextItem::editingFinished, this, [=]() {
        PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
        BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
        if (info.content != m_pContent->toPlainText())
        {
            updateBlackboard();
        }
    });

    bodyWidget->setLayout(pVLayout);

    if (blackboard.sz.isValid()) {
        resize(blackboard.sz);
    }

    return bodyWidget;
}

void BlackboardNode::updateBlackboard()
{
    PARAMS_INFO params = index().data(ROLE_PARAMS_NO_DESC).value<PARAMS_INFO>();
    BLACKBOARD_INFO info = params["blackboard"].value.value<BLACKBOARD_INFO>();
    if (m_pTitle)
    {
        info.title = m_pTitle->toPlainText();
    }
    if (m_pContent)
    {
        info.content = m_pContent->toPlainText();
    }
    info.sz = this->boundingRect().size();
    IGraphsModel *pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);
    pModel->updateBlackboard(index().data(ROLE_OBJID).toString(), QVariant::fromValue(info), subGraphIndex(), true);
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
    }
}

void BlackboardNode::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_bDragging)
    {
        QRectF hRect = m_headerWidget->sceneBoundingRect();
        QRectF bRect = m_bodyWidget->sceneBoundingRect();
        QPointF btopLeft = bRect.topLeft();
        qreal headerHeight = m_headerWidget->sceneBoundingRect().height();
        qreal h = m_headerWidget->sceneBoundingRect().height();
        qreal w = m_headerWidget->sceneBoundingRect().width();

        QPointF newBottomRight = event->scenePos();
        QPointF oldBottomRight = bRect.bottomRight();

        QRectF newBodyBr = QRectF(btopLeft, newBottomRight);
        QSizeF newBodySz = newBodyBr.size();

        zeno::log_info("newBodySz, width = {}, height = {}", newBodySz.width(), newBodySz.height());

        m_pContent->setData(GVKEY_SIZEHINT, newBodySz);
        ZGraphicsLayout::updateHierarchy(this);
        return;
    }

    ZenoNode::mouseMoveEvent(event);
}

void BlackboardNode::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    if (m_bDragging) {
        m_bDragging = false;
        ZGraphicsLayout::updateHierarchy(this);
        updateBlackboard();
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
    qreal manhatdist = offset.manhattanLength();
    return (manhatdist < ZenoStyle::dpiScaled(100));
}
