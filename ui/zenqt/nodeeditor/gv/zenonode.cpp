#include "zenonode.h"
#include "zenosubgraphscene.h"
#include "uicommon.h"
#include "control/common_id.h"
#include "nodeeditor/gv/zenoparamnameitem.h"
#include "nodeeditor/gv/zenoparamwidget.h"
#include "util/uihelper.h"
#include "model/GraphsTreeModel.h"
#include "model/graphsmanager.h"
#include "model/parammodel.h"
#include <zeno/utils/logger.h>
#include <zeno/utils/scope_exit.h>
#include "style/zenostyle.h"
#include "widgets/zveceditor.h"
#include "variantptr.h"
#include "curvemap/zcurvemapeditor.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "nodeeditor/gv/zenographseditor.h"
#include "util/log.h"
#include "zenosubgraphview.h"
#include "dialog/zenoheatmapeditor.h"
#include "nodeeditor/gv/zitemfactory.h"
#include "zvalidator.h"
#include "zenonewmenu.h"
#include "util/apphelper.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include "nodeeditor/gv/zgraphicstextitem.h"
#include "nodeeditor/gv/zenogvhelper.h"
#include "iotags.h"
#include "groupnode.h"
#include "dialog/zeditparamlayoutdlg.h"
#include "settings/zenosettingsmanager.h"
#include "nodeeditor/gv/zveceditoritem.h"
#include "nodeeditor/gv/zdictsocketlayout.h"
#include "zassert.h"
#include "widgets/ztimeline.h"
#include "socketbackground.h"
#include "statusgroup.h"
#include "statusbutton.h"


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_mainHeaderBg(nullptr)
    , m_topInputSockets(nullptr)
    , m_bottomOutputSockets(nullptr)
    , m_border(new QGraphicsRectItem)
    , m_NameItem(nullptr)
    , m_bError(false)
    , m_bEnableSnap(false)
    , m_bMoving(false)
    , m_bodyLayout(nullptr)
    , m_bUIInited(false)
    , m_inputsLayout(nullptr)
    , m_outputsLayout(nullptr)
    , m_groupNode(nullptr)
    , m_pStatusWidgets(nullptr)
    , m_bVisible(true)
    , m_NameItemTip(nullptr)
    , m_statusMarker(nullptr)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZenoNode::~ZenoNode()
{
}

void ZenoNode::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    if (isSelected())
    {
        _drawBorderWangStyle(painter);
    }
    zeno::NodeType type = static_cast<zeno::NodeType>(m_index.data(ROLE_NODETYPE).toInt());
    if (type == zeno::Node_Normal || type == zeno::SubInput || type == zeno::SubOutput) {
        painter->setRenderHint(QPainter::Antialiasing, true);
        QRectF r = m_headerWidget->boundingRect();
        qreal brWidth = ZenoStyle::scaleWidth(2);
        r.adjust(-brWidth / 2, -brWidth / 2, brWidth / 2, brWidth / 2);
        QPainterPath path;
        path.addRect(r);
        QPen pen(QColor(18,20,22), brWidth);
        pen.setJoinStyle(Qt::MiterJoin);
        painter->setPen(pen);
        painter->drawPath(path);
    }
}

void ZenoNode::_drawBorderWangStyle(QPainter* painter)
{
	//draw inner border
	painter->setRenderHint(QPainter::Antialiasing, true);
    QColor baseColor = /*m_bError ? QColor(200, 84, 79) : */QColor(255, 100, 0);
	QColor borderClr(baseColor);
	borderClr.setAlphaF(0.2);
    qreal innerBdrWidth = ZenoStyle::scaleWidth(6);
	QPen pen(borderClr, innerBdrWidth);
	pen.setJoinStyle(Qt::MiterJoin);
	painter->setPen(pen);

	QRectF rc = boundingRect();
	qreal offset = innerBdrWidth / 2; //finetune
	rc.adjust(-offset, -offset, offset, offset);
	QPainterPath path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
	painter->drawPath(path);

    //draw outter border
    qreal outterBdrWidth = ZenoStyle::scaleWidth(2);
    pen.setWidthF(outterBdrWidth);
    pen.setColor(baseColor);
	painter->setPen(pen);
    offset = outterBdrWidth;
    rc.adjust(-offset, -offset, offset, offset);
    path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
    painter->drawPath(path);
}

QRectF ZenoNode::boundingRect() const
{
    return _base::boundingRect();
}

int ZenoNode::type() const
{
    return Type;
}

void ZenoNode::initUI(ZenoSubGraphScene* pScene, const QModelIndex& subGIdx, const QModelIndex& index)
{
    ZASSERT_EXIT(index.isValid());
    m_index = QPersistentModelIndex(index);
    m_subGpIndex = QPersistentModelIndex(subGIdx);
    zeno::NodeType type = static_cast<zeno::NodeType>(m_index.data(ROLE_NODETYPE).toInt());

    m_topInputSockets = initVerticalSockets(true);
    m_mainHeaderBg = initMainHeaderBg();
    m_headerWidget = initHeaderWidget();
    m_bodyWidget = initBodyWidget(pScene);
    m_bottomOutputSockets = initVerticalSockets(false);

    ZGraphicsLayout* mainLayout = new ZGraphicsLayout(false);
    mainLayout->setDebugName("mainLayout");
    mainLayout->addLayout(m_topInputSockets);
    mainLayout->addItem(m_mainHeaderBg);
    mainLayout->addItem(m_headerWidget);
    mainLayout->addItem(m_bodyWidget);
    mainLayout->addLayout(m_bottomOutputSockets);

    mainLayout->setSpacing(0);
    setLayout(mainLayout);

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_NODE_NAME).toString();
    setPos(pos);

    // setPos will send geometry, but it's not supposed to happend during initialization.
    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    onCollaspeUpdated(bCollasped);

    //updateWhole();

    if (type == zeno::Node_Group) {
        setZValue(ZVALUE_BLACKBOARD);
    } else {
        //set no color for normal node(background)
        setColors(false, QColor(0, 0, 0, 0));
    }

    if (m_index.data(ROLE_NODE_DIRTY).toBool())
        onMarkDataChanged(true);

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();

    m_bUIInited = true;
    //onZoomed();
}

ZGraphicsLayout* ZenoNode::initVerticalSockets(bool bInput)
{
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM, nullptr);

    ZGraphicsLayout* pSocketLayout = new ZGraphicsLayout(true);
    pSocketLayout->addSpacing(-1);
    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        if (paramIdx.data(ROLE_ISINPUT).toBool() != bInput)
            continue;

        if (paramIdx.data(ROLE_SOCKET_TYPE) != zeno::PrimarySocket)
            continue;

        QSizeF szSocket(14, 14);
        ZenoSocketItem* socket = new ZenoSocketItem(paramIdx, ZenoStyle::dpiScaledSize(szSocket));
        socket->setBrush(QColor("#CECCCC"), QColor("#CECCCC"));
        pSocketLayout->addItem(socket);
        pSocketLayout->addSpacing(-1);

        QObject::connect(socket, &ZenoSocketItem::clicked, [=](bool bInput, zeno::LinkFunction lnkProp) {
            emit socketClicked(socket, lnkProp);
        });
    }
    return pSocketLayout;
}

ZLayoutBackground* ZenoNode::initMainHeaderBg()
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground;
    headerWidget->setRadius(10, 10, 10, 10);

    zeno::NodeType type = static_cast<zeno::NodeType>(m_index.data(ROLE_NODETYPE).toInt());

    QLinearGradient linearGradientH({ 0,0 }, { 100,100 });
    linearGradientH.setColorAt(0, Qt::black);
    linearGradientH.setColorAt(1, Qt::white);
    linearGradientH.setSpread(QGradient::PadSpread);
    headerWidget->setLinearGradient(linearGradientH);

    const QString& dispName = m_index.data(ROLE_NODE_DISPLAY_NAME).toString();
    ZASSERT_EXIT(!dispName.isEmpty(), headerWidget);
    const QString& name = m_index.data(ROLE_NODE_NAME).toString();
    const QString& nodeCls = m_index.data(ROLE_CLASS_NAME).toString();
    const QString& iconResPath = m_index.data(ROLE_NODE_DISPLAY_ICON).toString();

    QFont font2 = QApplication::font();
    font2.setPointSize(18);
    font2.setWeight(QFont::Normal);

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("Main Header HLayout");

    RoundRectInfo buttonShapeInfo;
    buttonShapeInfo.W = ZenoStyle::dpiScaled(24.);
    buttonShapeInfo.H = ZenoStyle::dpiScaled(64.);
    buttonShapeInfo.ltradius = ZenoStyle::dpiScaled(9.);
    buttonShapeInfo.lbradius = ZenoStyle::dpiScaled(9.);

    ZGraphicsLayout* pNameLayout = new ZGraphicsLayout(false);
    pNameLayout->setContentsMargin(5, 10, 5, 10);

    const QSizeF szIcon = ZenoStyle::dpiScaledSize(QSizeF(36, 36));
    if (!iconResPath.isEmpty())
    {
        ImageElement elem;
        elem.image = elem.imageHovered = elem.imageOn = elem.imageOnHovered = iconResPath;
        auto node_icon = new ZenoImageItem(elem, szIcon);
        pNameLayout->addItem(node_icon, Qt::AlignHCenter);
    }
    else {
        pNameLayout->addSpacing(36);
    }

    auto pCategoryItem = new ZSimpleTextItem(nodeCls);
    pCategoryItem->setBrush(QColor("#AB6E40"));
    font2.setPointSize(10);
    pCategoryItem->setFont(font2);
    pCategoryItem->updateBoundingRect();
    pCategoryItem->setAcceptHoverEvents(false);

    pNameLayout->addItem(pCategoryItem);

    pHLayout->addLayout(pNameLayout);

    buttonShapeInfo.ltradius = buttonShapeInfo.lbradius = 0.;
    buttonShapeInfo.rtradius = buttonShapeInfo.rbradius = ZenoStyle::dpiScaled(9.);

    StatusGroup* pStatusWidgets = new StatusGroup(buttonShapeInfo);
    bool bView = m_index.data(ROLE_NODE_ISVIEW).toBool();
    pStatusWidgets->setView(bView);
    connect(pStatusWidgets, &StatusGroup::toggleChanged, this, &ZenoNode::onOptionsBtnToggled);

    pHLayout->addItem(pStatusWidgets);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);

    //创建可以显示的名字组
    font2.setPointSize(20);

    auto nameItem = new ZEditableTextItem(name, headerWidget);
    nameItem->setDefaultTextColor(QColor("#CCCCCC"));
    nameItem->setTextLengthAsBounding(true);
    nameItem->setBackground(QColor(0, 0, 0, 0));
    nameItem->setFont(font2);
    qreal txtWid = nameItem->boundingRect().width();
    nameItem->setPos(-txtWid - 2, 14);

    connect(nameItem, &ZEditableTextItem::contentsChanged, this, [=]() {
        qreal ww = nameItem->textLength();
        nameItem->setPos(-ww - 2, 14);
    });

    return headerWidget;
}

ZLayoutBackground* ZenoNode::initHeaderWidget()
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground;
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, 10, headerBg.lb_radius, 0);
    qreal bdrWidth = ZenoStyle::dpiScaled(headerBg.border_witdh);
    headerWidget->setBorder(bdrWidth, headerBg.clr_border);

    ZASSERT_EXIT(m_index.isValid(), nullptr);

    zeno::NodeType type = static_cast<zeno::NodeType>(m_index.data(ROLE_NODETYPE).toInt());

    QColor clrHeaderBg;
    if (type == zeno::NoVersionNode)
        clrHeaderBg = QColor(83, 83, 85);
    else if (type == zeno::Node_SubgraphNode)
        clrHeaderBg = QColor("#1D5F51");
    else if (type == zeno::Node_AssetInstance || type == zeno::Node_AssetReference)
        clrHeaderBg = QColor("#d69b3c");
    else
        clrHeaderBg = headerBg.clr_normal;

    //headerWidget->setColors(headerBg.bAcceptHovers, clrHeaderBg, clrHeaderBg, clrHeaderBg);

    QLinearGradient linearGradientH({ 0,0 }, { 100,100 });
    linearGradientH.setColorAt(0, Qt::black);
    linearGradientH.setColorAt(1, Qt::white);
    linearGradientH.setSpread(QGradient::PadSpread);
    headerWidget->setLinearGradient(linearGradientH);

    headerWidget->setBorder(ZenoStyle::dpiScaled(headerBg.border_witdh), headerBg.clr_border);

    const QString& nodeCls = m_index.data(ROLE_CLASS_NAME).toString();
    const QString& name = m_index.data(ROLE_NODE_NAME).toString();
    const QString& dispName = m_index.data(ROLE_NODE_DISPLAY_NAME).toString();
    ZASSERT_EXIT(!dispName.isEmpty(), headerWidget);
    const QString& iconResPath = m_index.data(ROLE_NODE_DISPLAY_ICON).toString();

    const QString& category = m_index.data(ROLE_NODE_CATEGORY).toString();

    QFont font2 = QApplication::font();
    font2.setPointSize(18);
    font2.setWeight(QFont::Normal);

    auto nameItem = new ZSimpleTextItem(nodeCls, font2, QColor("#F1E9E9"));

    //qreal margin = ZenoStyle::dpiScaled(10);
    //pNameLayout->setContentsMargin(margin, margin, margin, margin);

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("Header HLayout");
    pHLayout->addSpacing(ZenoStyle::dpiScaled(16.));

    //icons

    //ZGraphicsLayout* pNameLayout = new ZGraphicsLayout(true);
    const QSizeF szIcon = ZenoStyle::dpiScaledSize(QSizeF(36, 36));
    if (!iconResPath.isEmpty())
    {
        ImageElement elem;
        elem.image = elem.imageHovered = elem.imageOn = elem.imageOnHovered = iconResPath;
        auto node_icon = new ZenoImageItem(elem, szIcon);
        pHLayout->addItem(node_icon, Qt::AlignVCenter);
    }
    else
    {
        pHLayout->addSpacing(szIcon.width());
    }

    //pNameLayout->addSpacing(-1);
    const qreal W_status = ZenoStyle::dpiScaled(22.);
    const qreal H_status = ZenoStyle::dpiScaled(50.);
    const qreal radius = ZenoStyle::dpiScaled(9.);

    RoundRectInfo buttonShapeInfo;
    buttonShapeInfo.W = ZenoStyle::dpiScaled(22.);
    buttonShapeInfo.H = ZenoStyle::dpiScaled(50.);
    buttonShapeInfo.rtradius = ZenoStyle::dpiScaled(9.);

    m_pStatusWidgets = new StatusGroup(buttonShapeInfo);
    bool bView = m_index.data(ROLE_NODE_ISVIEW).toBool();
    m_pStatusWidgets->setView(bView);
    connect(m_pStatusWidgets, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    //pHLayout->addLayout(pNameLayout);
    pHLayout->addSpacing(ZenoStyle::dpiScaled(5.));
    pHLayout->addItem(nameItem, Qt::AlignVCenter);

    //补充一些距离
    pHLayout->addSpacing(szIcon.width() + ZenoStyle::dpiScaled(20.));

    //pHLayout->addSpacing(100, QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed));
    pHLayout->addItem(m_pStatusWidgets, Qt::AlignRight);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    if (const GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_index.data(ROLE_GRAPH)))
    {
        m_pStatusWidgets->setEnabled(!pModel->isLocked());
        connect(pModel, &GraphModel::lockStatusChanged, this, [=]() {
            m_pStatusWidgets->setEnabled(!pModel->isLocked());
            for (auto layout : getSocketLayouts(true))
            {
                if (auto pControl = layout->control())
                    pControl->setEnabled(!pModel->isLocked());
            }
        });
    }

    m_statusMarker = new QGraphicsPolygonItem(headerWidget);
    QPolygonF points;
    points.append(QPointF(0, 0));
    points.append(QPointF(15, 0));
    points.append(QPointF(0, 15));
    m_statusMarker->setPolygon(points);
    m_statusMarker->setBrush(QBrush(QColor("#EAED4B")));
    m_statusMarker->setPen(Qt::NoPen);
    m_statusMarker->setPos(QPointF(0, 0));

    return headerWidget;
}

ZLayoutBackground* ZenoNode::initBodyWidget(ZenoSubGraphScene* pScene)
{
    ZLayoutBackground* bodyWidget = new ZLayoutBackground(this);
    const auto& bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, QColor("#2A2A2A"));

    qreal bdrWidth = ZenoStyle::dpiScaled(bodyBg.border_witdh);
    bodyWidget->setBorder(ZenoStyle::scaleWidth(2), QColor(18, 20, 22));

    m_bodyLayout = new ZGraphicsLayout(false);
    m_bodyLayout->setDebugName("Body Layout");
    qreal margin = ZenoStyle::dpiScaled(16);
    m_bodyLayout->setContentsMargin(margin, bdrWidth, 0, bdrWidth);

    ZASSERT_EXIT(m_index.isValid(), nullptr);
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM, nullptr);

    connect(paramsM, &ParamsModel::rowsInserted, this, &ZenoNode::onParamInserted);
    connect(paramsM, &ParamsModel::rowsAboutToBeRemoved, this, &ZenoNode::onViewParamAboutToBeRemoved);
    connect(paramsM, &ParamsModel::dataChanged, this, &ZenoNode::onParamDataChanged);
    connect(paramsM, &ParamsModel::rowsAboutToBeMoved, this, &ZenoNode::onViewParamAboutToBeMoved);
    connect(paramsM, &ParamsModel::rowsMoved, this, &ZenoNode::onViewParamsMoved);
    connect(paramsM, &ParamsModel::layoutAboutToBeChanged, this, &ZenoNode::onLayoutAboutToBeChanged);
    bool ret = connect(paramsM, &ParamsModel::layoutChanged, this, &ZenoNode::onLayoutChanged);

    m_inputsLayout = initSockets(paramsM, true, pScene);
    m_bodyLayout->addLayout(m_inputsLayout);

    m_outputsLayout = initSockets(paramsM, false, pScene);
    m_bodyLayout->addLayout(m_outputsLayout);

    m_bodyLayout->addSpacing(13, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

    bodyWidget->setLayout(m_bodyLayout);
    return bodyWidget;
}

void ZenoNode::onRunStateChanged()
{
    const NodeState& state = m_index.data(ROLE_NODE_RUN_STATE).value<NodeState>();
    this->onMarkDataChanged(state.bDirty);
    switch (state.runstatus) {
        case zeno::Node_Pending:
        {
            this->markError(false);
            break;
        }
        case zeno::Node_RunError:
        {
            this->markError(true);
            break;
        }
        case zeno::Node_Running:
        {
            this->markError(false);
            break;
        }
        case zeno::Node_RunSucceed:
        {
            this->markError(false);
            break;
        }
    }
}

void ZenoNode::onLayoutAboutToBeChanged()
{

}

QVector<ZSocketLayout*> ZenoNode::getSocketLayouts(bool bInput) const
{
    QVector<ZSocketLayout*> layouts;
    if (bInput) {
        for (int i = 0; i < m_inputsLayout->count(); i++) {
            ZGvLayoutItem* pItem = m_inputsLayout->itemAt(i);
            if (pItem->type == Type_Layout) {
                layouts.push_back(static_cast<ZSocketLayout*>(pItem->pLayout));
            }
            else {
                auto pBgItem = static_cast<ZLayoutBackground*>(pItem->pItem);
                layouts.push_back(static_cast<ZSocketLayout*>(pBgItem->layout()));
            }
        }
    }
    else {
        for (int i = 0; i < m_outputsLayout->count(); i++) {
            ZGvLayoutItem* pItem = m_outputsLayout->itemAt(i);
            if (pItem->type == Type_Layout) {
                layouts.push_back(static_cast<ZSocketLayout*>(pItem->pLayout));
            }
            else {
                auto pBgItem = static_cast<ZLayoutBackground*>(pItem->pItem);
                layouts.push_back(static_cast<ZSocketLayout*>(pBgItem->layout()));
            }
        }
    }
    return layouts;
}

void ZenoNode::onLayoutChanged()
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    m_inputsLayout->clear();
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);

    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        if (!paramIdx.data(ROLE_ISINPUT).toBool())
            continue;
        m_inputsLayout->addItem(addSocket(paramIdx, true, pScene));
    }

    m_outputsLayout->clear();
    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        if (paramIdx.data(ROLE_ISINPUT).toBool())
            continue;
        m_outputsLayout->addItem(addSocket(paramIdx, false, pScene));
    }

    updateWhole();
}

ZGraphicsLayout* ZenoNode::initCustomParamWidgets()
{
    return nullptr;
}

QPersistentModelIndex ZenoNode::subGraphIndex() const
{
    return m_subGpIndex;
}

void ZenoNode::onNameUpdated(const QString& newName)
{
    ZASSERT_EXIT(m_NameItem);
    if (m_NameItem)
    {
        const QString& name = m_index.data(ROLE_NODE_NAME).toString();
        m_NameItem->setText(name);
        ZGraphicsLayout::updateHierarchy(m_NameItem);
    }
}

ZSocketLayout* ZenoNode::getSocketLayout(bool bInput, const QString& name)
{
    QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
    for (int i = 0; i < layouts.size(); i++)
    {
        QModelIndex idx = layouts[i]->viewSocketIdx();
        QString sockName = idx.data(ROLE_PARAM_NAME).toString();
        if (sockName == name)
            return layouts[i];
    }
    return nullptr;
}

ZSocketLayout* ZenoNode::getSocketLayout(bool bInput, int idx)
{
    ZGraphicsLayout* groupLayout = bInput ? m_inputsLayout : m_outputsLayout;
    ZGvLayoutItem* pItem = groupLayout->itemAt(idx);
    ZASSERT_EXIT(pItem, nullptr);
    if (pItem->type == Type_Layout) {
        return static_cast<ZSocketLayout*>(pItem->pLayout);
    }
    else {
        auto pBgItem = static_cast<ZLayoutBackground*>(pItem->pItem);
        return static_cast<ZSocketLayout*>(pBgItem->layout());
    }
}

bool ZenoNode::removeSocketLayout(bool bInput, const QString& name)
{
    if (bInput)
    {
        QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
        for (int i = 0; i < layouts.size(); i++)
        {
            QModelIndex idx = layouts[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
            {
                m_inputsLayout->removeElement(i);
                return false;
            }
        }
    }
    else
    {
        QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
        for (int i = 0; i < layouts.size(); i++)
        {
            QModelIndex idx = layouts[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
            {
                m_outputsLayout->removeElement(i);
                return false;
            }
        }
    }
    return false;
}

void ZenoNode::onParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (roles.isEmpty())
        return;

    if (!m_index.isValid())
        return;

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    if (!paramsM)
        return;

    QModelIndex paramIdx = topLeft;

    for (int role : roles)
    {
        if (role != ROLE_PARAM_NAME
            && role != ROLE_PARAM_VALUE
            && role != ROLE_PARAM_CONTROL
            && role != ROLE_PARAM_CTRL_PROPERTIES
            && role != ROLE_PARAM_TOOLTIP)
            return;

        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
        ZASSERT_EXIT(pScene);

        const bool bInput = paramIdx.data(ROLE_ISINPUT).toBool();
        const QString& paramName = paramIdx.data(ROLE_PARAM_NAME).toString();
        const auto paramCtrl = paramIdx.data(ROLE_PARAM_CONTROL).toInt();
        const zeno::ParamType paramType = (zeno::ParamType)paramIdx.data(ROLE_PARAM_TYPE).toInt();

        if (role == ROLE_PARAM_NAME || role == ROLE_PARAM_TOOLTIP)
        {
            QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
            for (int i = 0; i < layouts.size(); i++)
            {
                ZSocketLayout* pSocketLayout = layouts[i];
                QModelIndex socketIdx = pSocketLayout->viewSocketIdx();
                if (socketIdx == paramIdx)
                {
                    if (role == ROLE_PARAM_NAME)
                        pSocketLayout->updateSockName(paramName);   //only update name on control.
                    else if (role == ROLE_PARAM_TOOLTIP)
                        pSocketLayout->updateSockNameToolTip(paramIdx.data(ROLE_PARAM_TOOLTIP).toString());
                    break;
                }
            }
            updateWhole();
            return;
        }

        if (bInput)
        {
            ZSocketLayout* pControlLayout = getSocketLayout(true, paramName);
            QGraphicsItem* pControl = nullptr;
            if (pControlLayout)
                pControl = pControlLayout->control();

            switch (role)
            {
            case ROLE_PARAM_CONTROL:
            {
                const auto oldCtrl = pControl ? pControl->data(GVKEY_CONTROL).toInt() : zeno::NullControl;
                if (paramCtrl != oldCtrl)
                {
                    QGraphicsItem* pNewControl = initSocketWidget(pScene, paramIdx);
                    pControlLayout->setControl(pNewControl);
                    if (pNewControl)
                        pNewControl->setVisible(pControlLayout->socketItem()->sockStatus() != ZenoSocketItem::STATUS_CONNECTED);
                    pControl = pNewControl;
                    updateWhole();
                }
                //set value on pControl.
                const QVariant& deflValue = paramIdx.data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, paramType, deflValue, pScene);
                break;
            }
            case ROLE_PARAM_VALUE:
            {
                const QVariant& deflValue = paramIdx.data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, paramType, deflValue, pScene);
                if (zeno::Param_Vec4f == paramType ||
                    zeno::Param_Vec3f == paramType ||
                    zeno::Param_Vec2f == paramType ||
                    zeno::Param_Float == paramType)
                {
                    if (QGraphicsProxyWidget* pWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pControl))
                    {
                        if (ZVecEditorItem* pEditor = qobject_cast<ZVecEditorItem*>(pWidget))
                        {
                            QVariant newVal = deflValue;
                            bool bKeyFrame = AppHelper::getCurveValue(newVal);
                            if (bKeyFrame)
                                pEditor->setVec(newVal, true, pScene);
                            QVector<QString> properties = AppHelper::getKeyFrameProperty(deflValue);
                            pEditor->updateProperties(properties);
                        }
                    }
                    else if (QGraphicsTextItem* pTextItem = qgraphicsitem_cast<QGraphicsTextItem*>(pControl))
                    {
                        QVariant newVal = deflValue;
                        bool bKeyFrame = AppHelper::getCurveValue(newVal);
                        if (bKeyFrame)
                            pTextItem->setPlainText(UiHelper::variantToString(newVal));
                        QVector<QString> properties = AppHelper::getKeyFrameProperty(deflValue);
                        pTextItem->setProperty(g_setKey, properties.first());
                    }
                }
                break;
            }
            case ROLE_PARAM_CTRL_PROPERTIES:
            {
                QVariant value = paramIdx.data(ROLE_PARAM_CTRL_PROPERTIES);
                ZenoGvHelper::setCtrlProperties(pControl, value);
                const QVariant& deflValue = paramIdx.data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, paramType, deflValue, pScene);
                break;
            }
            }
        }
        else
        {
        }
    }
}

void ZenoNode::onParamInserted(const QModelIndex& parent, int first, int last)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);

    for (int r = first; r <= last; r++)
    {
        QModelIndex paramIdx = paramsM->index(r, 0, parent);
        bool bInput = paramIdx.data(ROLE_ISINPUT).toBool();
        ZGraphicsLayout* pSocketsLayout = bInput ? m_inputsLayout : m_outputsLayout;
        pSocketsLayout->addItem(addSocket(paramIdx, bInput, pScene));
        updateWhole();
    }
}

void ZenoNode::onViewParamAboutToBeMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row)
{

}

void ZenoNode::onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int destRow)
{
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(sender());
    ZASSERT_EXIT(paramsM);

    //TODO: output case
    m_inputsLayout->moveItem(start, destRow);
    updateWhole();
}

void ZenoNode::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (!parent.isValid())
    {
        //remove all component.
        m_inputsLayout->clear();
        m_outputsLayout->clear();
        m_params.clear();
        //m_inSockets.clear();
        //m_outSockets.clear();
        return;
    }

    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    for (int r = first; r <= last; r++)
    {
        QModelIndex viewParamIdx = paramsM->index(r, 0, parent);
        const int paramCtrl = viewParamIdx.data(ROLE_PARAM_CONTROL).toInt();
        bool bInput = viewParamIdx.data(ROLE_ISINPUT).toBool();

        const QString& paramName = viewParamIdx.data(ROLE_PARAM_NAME).toString();
        ZSocketLayout* pSocketLayout = getSocketLayout(bInput, paramName);
        removeSocketLayout(bInput, paramName);

        ZASSERT_EXIT(pSocketLayout);
        ZGraphicsLayout* pParentLayout = pSocketLayout->parentLayout();
        pParentLayout->removeLayout(pSocketLayout);
        updateWhole();
    }
}

void ZenoNode::focusOnNode(const QModelIndex& nodeIdx)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
    ZASSERT_EXIT(pScene && !pScene->views().isEmpty());
    if (_ZenoSubGraphView* pView = qobject_cast<_ZenoSubGraphView*>(pScene->views().first()))
    {
        ZASSERT_EXIT(nodeIdx.isValid());
        pView->focusOn(nodeIdx.data(ROLE_NODE_NAME).toString(), QPointF(), false);
    }
}

ZGraphicsLayout* ZenoNode::initSockets(ParamsModel* pModel, const bool bInput, ZenoSubGraphScene* pScene)
{
    ZASSERT_EXIT(pModel, nullptr);

    ZGraphicsLayout* pSocketsLayout = new ZGraphicsLayout(false);
    pSocketsLayout->setSpacing(5);

    for (int r = 0; r < pModel->rowCount(); r++)
    {
        const QModelIndex& paramIdx = pModel->index(r, 0);
        if (paramIdx.data(ROLE_ISINPUT).toBool() != bInput)
            continue;
        pSocketsLayout->addItem(addSocket(paramIdx, bInput, pScene));
    }
    return pSocketsLayout;
}

SocketBackgroud* ZenoNode::addSocket(const QModelIndex& paramIdx, bool bInput, ZenoSubGraphScene* pScene)
{
    SocketBackgroud* pBackground = new SocketBackgroud(bInput);

    QPersistentModelIndex perSockIdx = paramIdx;

    CallbackForSocket cbSocket;
    cbSocket.cbOnSockClicked = [=](ZenoSocketItem* pSocketItem, zeno::LinkFunction lnkProp) {
        emit socketClicked(pSocketItem, lnkProp);
    };
    cbSocket.cbOnSockLayoutChanged = [=]() {
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    };
    cbSocket.cbActionTriggered = [=](QAction* pAction, const QModelIndex& socketIdx) {
        QString text = pAction->text();
        if (pAction->text() == tr("Delete Net Label")) {
            //DEPRECATED
        }
    };

    const QString& sockName = paramIdx.data(ROLE_PARAM_NAME).toString();
    const zeno::ParamControl ctrl = (zeno::ParamControl)paramIdx.data(ROLE_PARAM_CONTROL).toInt();
    const zeno::ParamType type = (zeno::ParamType)paramIdx.data(ROLE_PARAM_TYPE).toInt();
    const QVariant& deflVal = paramIdx.data(ROLE_PARAM_VALUE);
    const PARAM_LINKS& links = paramIdx.data(ROLE_LINKS).value<PARAM_LINKS>();
    int sockProp = paramIdx.data(ROLE_PARAM_SOCKPROP).toInt();

    zeno::NodeType nodetype = static_cast<zeno::NodeType>(m_index.data(ROLE_NODETYPE).toInt());
    bool bSocketEnable = true;
    if (SOCKPROP_LEGACY == paramIdx.data(ROLE_PARAM_SOCKPROP) || nodetype == zeno::NoVersionNode || pScene->getGraphModel()->isLocked())
    {
        bSocketEnable = false;
    }

    ZSocketLayout* pMiniLayout = nullptr;
    if ((type == zeno::Param_Dict || type == zeno::Param_List) && 
        ctrl != zeno::NoMultiSockPanel) {
        pMiniLayout = new ZDictSocketLayout(paramIdx, bInput, pBackground);
    }
    else {
        pMiniLayout = new ZSocketLayout(paramIdx, bInput, pBackground);
        qreal margin = ZenoStyle::dpiScaled(16);
        if (bInput)
            pMiniLayout->setContentsMargin(0, 0, 0, margin);
    }

    pMiniLayout->initUI(cbSocket);
    pMiniLayout->setDebugName(sockName);

    if (bInput)
    {
        QGraphicsItem* pSocketControl = initSocketWidget(pScene, paramIdx);
        pMiniLayout->setControl(pSocketControl);
        if (pSocketControl) {
            pSocketControl->setVisible(links.isEmpty());
            pSocketControl->setEnabled(bSocketEnable);
        }
    }

    pBackground->setLayout(pMiniLayout);
    return pBackground;
}

Callback_OnButtonClicked ZenoNode::registerButtonCallback(const QModelIndex& paramIdx)
{
    return []() {

    };
}

QGraphicsItem* ZenoNode::initSocketWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx)
{
    const QPersistentModelIndex perIdx(paramIdx);

    zeno::ParamType sockType = (zeno::ParamType)paramIdx.data(ROLE_PARAM_TYPE).toInt();
    zeno::ParamControl ctrl = (zeno::ParamControl)paramIdx.data(ROLE_PARAM_CONTROL).toInt();
    bool bFloat = UiHelper::isFloatType(sockType);
    auto cbUpdateSocketDefl = [=](QVariant newValue) {
        const auto& oldVal = paramIdx.data(ROLE_PARAM_VALUE);
        if (oldVal == newValue)
            return;
        if (bFloat)
        {
            if (!AppHelper::updateCurve(oldVal, newValue))
            {
                onParamDataChanged(perIdx, perIdx, QVector<int>() << ROLE_PARAM_VALUE);
                return;
            }
        }
        if (!UiHelper::qIndexSetData(perIdx, newValue, ROLE_PARAM_VALUE))
        {
            QMessageBox::warning(nullptr, tr("Warning"), tr("Set data failed!"));
        }
    };
    auto cbUpdateSocketDefldWithSlider = [=](QVariant newValue) {
        //TODO: need this anymore?
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    const QVariant& deflVal = paramIdx.data(ROLE_PARAM_VALUE);
    zeno::ControlProperty ctrlProps = paramIdx.data(ROLE_PARAM_CTRL_PROPERTIES).value<zeno::ControlProperty>();

    auto cbGetIndexData = [=]() -> QVariant { 
        return perIdx.data(ROLE_PARAM_VALUE);
    };

    auto cbGetZsgDir = []() {
        QString path = zenoApp->graphsManager()->zsgPath();
        if (path.isEmpty())
            return QString("");
        QFileInfo fi(path);
        QString dirPath = fi.dir().path();
        return dirPath;
    };

    CallbackCollection cbSet;
    cbSet.cbEditFinished = cbUpdateSocketDefl;
    cbSet.cbEditFinishedWithSlider = cbUpdateSocketDefldWithSlider;
    cbSet.cbSwitch = cbSwith;
    cbSet.cbGetIndexData = cbGetIndexData;
    cbSet.cbGetZsgDir = cbGetZsgDir;
    cbSet.cbBtnOnClicked = registerButtonCallback(paramIdx);

    QVariant newVal = deflVal;
    if (bFloat)
    {
        AppHelper::getCurveValue(newVal);
    }
    QGraphicsItem* pControl = zenoui::createItemWidget(newVal, ctrl, sockType, cbSet, scene, ctrlProps);

    if (bFloat) {
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, pControl);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, pControl);
        connect(timeline, &ZTimeline::sliderValueChanged, this, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, paramIdx.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, this, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, paramIdx.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }

    return pControl;
}

void ZenoNode::onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded, const QString keyName)
{
    ZenoSocketItem* pSocket = getSocketItem(paramIdx, keyName);
    if (pSocket == nullptr)
        return;

    QModelIndex idx = pSocket->paramIndex();
    // the removal of links from socket is executed before the removal of link itself.
    PARAM_LINKS links = idx.data(ROLE_LINKS).value<PARAM_LINKS>();
    if (bAdded) {
        pSocket->setSockStatus(ZenoSocketItem::STATUS_CONNECTED);
    } else {
        if (links.isEmpty())
            pSocket->setSockStatus(ZenoSocketItem::STATUS_NOCONN);
    }

    if (bInput)
    {
        QString sockName = paramIdx.data(ROLE_PARAM_NAME).toString();
        // special case, we need to show the button param.
        if (this->nodeClass() == "GenerateCommands" && sockName == "source")
            return;

        ZSocketLayout* pSocketLayout = getSocketLayout(bInput, sockName);
        if (pSocketLayout && pSocketLayout->control())
        {
            pSocketLayout->control()->setVisible(!bAdded);
            updateWhole();
        }
    }
}

void ZenoNode::markError(bool isError)
{
    m_bError = isError;
    ZASSERT_EXIT(m_headerWidget);
    QColor clrHeaderBg;
    if (m_bError)
        clrHeaderBg = QColor(200, 84, 79);
    else if (m_index.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode)
        clrHeaderBg = QColor("#1D5F51");
    else
        clrHeaderBg = m_renderParams.headerBg.clr_normal;
    m_headerWidget->setColors(false, clrHeaderBg, QColor(), QColor());
    update();
}


QVector<ZenoSocketItem*> ZenoNode::getSocketItems(bool bInput) const
{
    QVector<ZenoSocketItem*> sockets;
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped)
    {
        ZGraphicsLayout* socketsLayout = bInput ? m_topInputSockets : m_bottomOutputSockets;
        for (int i = 0; i < socketsLayout->count(); i++)
        {
            ZGvLayoutItem* layoutitem = socketsLayout->itemAt(i);
            if (layoutitem->type == Type_Item)
            {
                if (ZenoSocketItem* pSocket = qgraphicsitem_cast<ZenoSocketItem*>(layoutitem->pItem))
                {
                    sockets.append(pSocket);
                }
            }
        }
    }
    else
    {
        auto socks = getSocketLayouts(bInput);
        for (ZSocketLayout* sock : socks)
        {
            ZenoSocketItem* pSocketItem = sock->socketItem();
            sockets.append(pSocketItem);
        }
    }
    return sockets;
}

ZenoSocketItem* ZenoNode::getSocketItem(const QModelIndex& sockIdx, const QString keyName)
{
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    const bool bInput = sockIdx.data(ROLE_ISINPUT).toBool();
    if (bCollasped)
    {
        ZGraphicsLayout* socketsLayout = bInput ? m_topInputSockets : m_bottomOutputSockets;
        for (int i = 0; i < socketsLayout->count(); i++)
        {
            ZGvLayoutItem* layoutitem = socketsLayout->itemAt(i);
            if (layoutitem->type == Type_Item)
            {
                if (ZenoSocketItem* pSocket = qgraphicsitem_cast<ZenoSocketItem*>(layoutitem->pItem))
                {
                    if (pSocket->paramIndex() == sockIdx)
                        return pSocket;
                }
            }
        }
        return nullptr;
    }
    else
    {
        for (ZSocketLayout* socklayout : getSocketLayouts(true))
        {
            if (ZenoSocketItem* pItem = socklayout->socketItemByIdx(sockIdx, keyName))
            {
                return pItem;
            }
        }
        for (ZSocketLayout* socklayout : getSocketLayouts(false))
        {
            if (ZenoSocketItem* pItem = socklayout->socketItemByIdx(sockIdx, keyName))
                return pItem;
        }
        return nullptr;
    }
}

ZenoSocketItem* ZenoNode::getNearestSocket(const QPointF& pos, bool bInput)
{
    ZenoSocketItem* pItem = nullptr;
    float minDist = std::numeric_limits<float>::max();
    for (ZenoSocketItem* pSocketItem : getSocketItems(bInput))
    {
        if (!pSocketItem || !pSocketItem->isEnabled())
            continue;

        QPointF sockPos = pSocketItem->center();
        QPointF offset = sockPos - pos;
        float dist = std::sqrt(offset.x() * offset.x() + offset.y() * offset.y());
        if (dist < minDist)
        {
            minDist = dist;
            pItem = pSocketItem;
        }
    }
    return pItem;
}

QModelIndex ZenoNode::getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const
{
    for (auto param : m_params)
    {
        if (bSocketText) {
            if (param.second.param_name == uiitem)
                return param.second.viewidx;
        }
        else {
            if (param.second.param_control == uiitem)
                return param.second.viewidx;
        }
    }
    for (ZSocketLayout* sock : getSocketLayouts(true))
    {
        if (bSocketText) {
            if (sock->socketText() == uiitem) {
                return sock->viewSocketIdx();
            }
        }
        else {
            if (sock->control() == uiitem) {
                return sock->viewSocketIdx();
            }
        }
    }
    for (ZSocketLayout* sock : getSocketLayouts(false))
    {
        if (bSocketText) {
            if (sock->socketText() == uiitem) {
                return sock->viewSocketIdx();
            }
        }
        else {
            if (sock->control() == uiitem) {
                return sock->viewSocketIdx();
            }
        }
    }
    return QModelIndex();
}

QPointF ZenoNode::getSocketPos(const QModelIndex& sockIdx, const QString keyName)
{
    ZASSERT_EXIT(sockIdx.isValid(), QPointF());

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    const bool bInput = sockIdx.data(ROLE_ISINPUT).toBool();
    if (bCollasped)
    {
        if (ZenoSocketItem* pSocket = getSocketItem(sockIdx, ""))
        {
            return pSocket->center();
        }
        zeno::log_warn("socket pos error");
        return QPointF(0, 0);
    }
    else
    {
        for (ZSocketLayout* socklayout : getSocketLayouts(bInput))
        {
            bool exist = false;
            QPointF pos = socklayout->getSocketPos(sockIdx, keyName, exist);
            if (exist)
                return pos;
        }
        zeno::log_warn("socket pos error");
        return QPointF(0, 0);
    }
}

QString ZenoNode::nodeId() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_NODE_NAME).toString();
}

QString ZenoNode::nodeClass() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_CLASS_NAME).toString();
}

QString ZenoNode::nodeUuid() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return "";
}

QPointF ZenoNode::nodePos() const
{
    ZASSERT_EXIT(m_index.isValid(), QPointF());
    return m_index.data(ROLE_OBJPOS).toPointF();
}

void ZenoNode::updateNodePos(const QPointF &pos, bool enableTransaction) 
{
    //this is a blackboard
    QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();
    if (oldPos == pos)
        return;
    //TODO
}

void ZenoNode::onUpdateParamsNotDesc()
{
}

void ZenoNode::onMarkDataChanged(bool bDirty)
{
    m_statusMarker->setVisible(bDirty);
    update();
}

void ZenoNode::setMoving(bool isMoving)
{
    m_bMoving = isMoving;
}

bool ZenoNode::isMoving() {
    return m_bMoving;
}

void ZenoNode::onZoomed()
{
    m_pStatusWidgets->onZoomed();
    bool bVisible = true;
    if (editor_factor < 0.2) {
        bVisible = false;
    }
    if (m_bVisible != bVisible) {
        m_bVisible = bVisible;
        for (auto item : getSocketLayouts(true)) {
            if (!item->isHide())
                item->setVisible(bVisible);
        }
        for (auto item : getSocketLayouts(false)) {
            if (!item->isHide())
                item->setVisible(bVisible);
        }
        for (auto it = m_params.begin(); it != m_params.end(); it++) {
            if (it->second.param_control)
                it->second.param_control->setVisible(bVisible);
            if (it->second.param_name)
                it->second.param_name->setVisible(bVisible);
        }
        //if (m_NameItem) {
        //    m_NameItem->setVisible(bVisible);
        //}
        if (m_bVisible) {
            updateWhole();
        }
    }

    if (editor_factor < 0.1 || editor_factor > 0.2) 
    {
        if (m_NameItemTip) 
        {
            delete m_NameItemTip;
            m_NameItemTip = nullptr;
        }
    }
    else if (m_NameItemTip == nullptr) 
    {
        const QString& nodeCls = m_index.data(ROLE_CLASS_NAME).toString();
        m_NameItemTip = new ZSimpleTextItem(nodeCls, this);
        m_NameItemTip->setBrush(QColor("#FFFFFF"));
        m_NameItemTip->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        m_NameItemTip->show();
    }
    if (m_NameItemTip) 
    {
        //if (m_NameItemTip->text() != m_NameItem->toPlainText())
        //    m_NameItemTip->setText(m_NameItem->toPlainText());
        //m_NameItemTip->setPos(QPointF(m_NameItem->pos().x(), -ZenoStyle::scaleWidth(36)));
    }
    if (m_bodyWidget)
        m_bodyWidget->setBorder(ZenoStyle::scaleWidth(2), QColor(18, 20, 22));
}

void ZenoNode::setGroupNode(GroupNode *pNode) 
{
    m_groupNode = pNode;
}

GroupNode *ZenoNode::getGroupNode() 
{
    return m_groupNode;
}

void ZenoNode::addParam(const _param_ctrl &param) 
{
    m_params[param.param_name->text()] = param;
}

void ZenoNode::setSelected(bool selected)
{
    _base::setSelected(selected);
}

bool ZenoNode::sceneEventFilter(QGraphicsItem *watched, QEvent *event) {
    return _base::sceneEventFilter(watched, event);
}

bool ZenoNode::sceneEvent(QEvent *event)
{
    return _base::sceneEvent(event);
}

ZenoGraphsEditor* ZenoNode::getEditorViewByViewport(QWidget* pWidget)
{
    QWidget* p = pWidget;
    while (p)
    {
        if (ZenoGraphsEditor* pEditor = qobject_cast<ZenoGraphsEditor*>(p))
            return pEditor;
        p = p->parentWidget();
    }
    return nullptr;
}

void ZenoNode::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    auto graphsMgr = zenoApp->graphsManager();
    QPointF pos = event->pos();
    if (m_index.data(ROLE_NODETYPE) == zeno::Node_SubgraphNode)
    {
        scene()->clearSelection();
        this->setSelected(true);

        QMenu *nodeMenu = new QMenu;
        QAction *pCopy = new QAction("Copy");
        QAction *pDelete = new QAction("Delete");

        connect(pDelete, &QAction::triggered, this, [=]() {
            //pGraphsModel->removeNode(m_index.data(ROLE_NODE_NAME).toString(), m_subGpIndex, true);
        });

        nodeMenu->addAction(pCopy);
        nodeMenu->addAction(pDelete);

        QAction* propDlg = new QAction(tr("Custom Param"));
        nodeMenu->addAction(propDlg);
        connect(propDlg, &QAction::triggered, this, [=]() {
            ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(ROLE_PARAMS));

            ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
            ZASSERT_EXIT(pScene && !pScene->views().isEmpty());
            if (_ZenoSubGraphView* pView = qobject_cast<_ZenoSubGraphView*>(pScene->views().first()))
            {
                ZEditParamLayoutDlg dlg(paramsM->customParamModel(), pView);
                if (QDialog::Accepted == dlg.exec())
                {
                    zeno::ParamsUpdateInfo info = dlg.getEdittedUpdateInfo();
                    paramsM->batchModifyParams(info);
                }
            }
        });

        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else if (m_index.data(ROLE_NODETYPE) == zeno::Node_AssetInstance)
    {
        GraphModel* pSubgGraphM = m_index.data(ROLE_SUBGRAPH).value<GraphModel*>();
        ZASSERT_EXIT(pSubgGraphM);
        bool bLocked = pSubgGraphM->isLocked();
        QMenu* nodeMenu = new QMenu;
        QAction* pLock = new QAction(bLocked ? tr("UnLock") : tr("Lock"));
        nodeMenu->addAction(pLock);
        connect(pLock, &QAction::triggered, this, [=]() {
            pSubgGraphM->setLocked(!bLocked);
        });
        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else if (m_index.data(ROLE_CLASS_NAME).toString() == "BindMaterial")
    {
#if 0
        QAction* newSubGraph = new QAction(tr("Create Material Subgraph"));
        connect(newSubGraph, &QAction::triggered, this, [=]() {
            NodeParamModel* pNodeParams = QVariantPtr<NodeParamModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
            ZASSERT_EXIT(pNodeParams);
            const auto& paramIdx = pNodeParams->getParam(PARAM_INPUT, "mtlid");
            ZASSERT_EXIT(paramIdx.isValid());
            QString mtlid = paramIdx.data(ROLE_PARAM_VALUE).toString();
            if (!pGraphsModel->newMaterialSubgraph(m_subGpIndex, mtlid, this->pos() + QPointF(800, 0)))
                QMessageBox::warning(nullptr, tr("Info"), tr("Create material subgraph '%1' failed.").arg(mtlid));
        });
        QMenu *nodeMenu = new QMenu;
        nodeMenu->addAction(newSubGraph);
        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
#endif
    }
    else
    {
        //zeno::NodeCates cates = graphsMgr->getCates();
        //pos = event->screenPos();
        //ZenoNewnodeMenu *menu = new ZenoNewnodeMenu(m_subGpIndex, cates, pos);
        //menu->setEditorFocus();
        //menu->exec(pos.toPoint());
        //menu->deleteLater();
    }
}

void ZenoNode::focusOutEvent(QFocusEvent* event)
{
    _base::focusOutEvent(event);
}

bool ZenoNode::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_NameItem)
    {
        if ((event->type() == QEvent::InputMethod || event->type() == QEvent::KeyPress) && m_NameItem->textInteractionFlags() == Qt::TextEditable)
        {
            bool bDelete = false;
            if (event->type() == QEvent::KeyPress)
            {
                QKeyEvent* pKeyEvent = static_cast<QKeyEvent*>(event);
                int key = pKeyEvent->key();
                if (key == Qt::Key_Delete || key == Qt::Key_Backspace)
                    bDelete = true;
            }
            if (!bDelete)
            {
                QString name = m_index.data(ROLE_CLASS_NAME).toString();
                QColor color = QColor(255, 255, 255);
                QColor textColor = m_NameItem->defaultTextColor();
                if (textColor != color)
                {
                    m_NameItem->setDefaultTextColor(color);
                }

                if (m_NameItem->toPlainText() == name)
                {
                    m_NameItem->setText("");
                }

                if (m_NameItem->textInteractionFlags() != Qt::TextEditorInteraction)
                {
                    m_NameItem->setTextInteractionFlags(Qt::TextEditorInteraction);
                }
            }
        }
        else if (event->type() == QEvent::KeyRelease && m_NameItem->textInteractionFlags() == Qt::TextEditorInteraction)
        {
            QString text = m_NameItem->toPlainText();
            if (text.isEmpty())
            {
                QString name = m_index.data(ROLE_CLASS_NAME).toString();
                m_NameItem->setText(name);
                m_NameItem->setTextInteractionFlags(Qt::TextEditable);
                m_NameItem->setDefaultTextColor(QColor(255, 255, 255, 40));
            }
        }
    }
    return _base::eventFilter(obj, event);
}

void ZenoNode::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mousePressEvent(event);
}

void ZenoNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    if (const GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_index.data(ROLE_GRAPH)))
    {
        if (pModel->isLocked())
            return;
    }
    _base::mouseDoubleClickEvent(event);
    QList<QGraphicsItem*> items = scene()->items(event->scenePos());
    if (items.contains(m_NameItem))
    {
        QString name = m_index.data(ROLE_CLASS_NAME).toString();
        if (name == m_NameItem->toPlainText())
        {
            m_NameItem->setTextInteractionFlags(Qt::TextEditable);
            m_NameItem->setDefaultTextColor(QColor(255, 255, 255, 40));
        }
        else
        {
            m_NameItem->setTextInteractionFlags(Qt::TextEditorInteraction);
            m_NameItem->setDefaultTextColor(QColor(255, 255, 255));
        }
        m_NameItem->setFocus();
    }
    else if (items.contains(m_headerWidget) || items.contains(m_mainHeaderBg))
    {
        onCollaspeBtnClicked();
    }
    else if (items.contains(m_bodyWidget))
    {
        const QModelIndex& nodeIdx = index();
        zeno::NodeType type = (zeno::NodeType)nodeIdx.data(ROLE_NODETYPE).toInt();
        if (type == zeno::Node_SubgraphNode || type == zeno::Node_AssetInstance)
        {
            //fork and expand asset graph
            ZenoGraphsEditor* pEditor = getEditorViewByViewport(event->widget());
            if (pEditor)
            {
                pEditor->onPageActivated(nodeIdx);
            }
        }
        else if (type == zeno::Node_AssetReference)
        {
            ZenoGraphsEditor* pEditor = getEditorViewByViewport(event->widget());
            if (pEditor)
            {
                QString assetName = nodeIdx.data(ROLE_CLASS_NAME).toString();
                pEditor->activateTab({ assetName });
            }
        }
        // for temp support to show handler via transform node
        else if (nodeClass().contains("TransformPrimitive"))
        {
            QVector<DisplayWidget*> views = zenoApp->getMainWindow()->viewports();
            for (auto pDisplay : views)
            {
                ZASSERT_EXIT(pDisplay);
                pDisplay->changeTransformOperation(nodeId());
            }
        }
    }
}

void ZenoNode::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseMoveEvent(event);
}

void ZenoNode::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
    if (m_bMoving)
    {
        m_bMoving = false;
        QPointF newPos = this->scenePos();
        QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();
        if (newPos != oldPos)
        {
            UiHelper::qIndexSetData(m_index, m_lastMoving, ROLE_OBJPOS);

            emit inSocketPosChanged();
            emit outSocketPosChanged();
            //emit nodePosChangedSignal();

            m_lastMoving = QPointF();

            //other selected items also need update model data
            for (QGraphicsItem *item : this->scene()->selectedItems()) {
                if (item == this || !dynamic_cast<ZenoNode*>(item))
                    continue;
                ZenoNode *pNode = dynamic_cast<ZenoNode *>(item);
                UiHelper::qIndexSetData(pNode->index(), pNode->scenePos(), ROLE_OBJPOS);
            }
        }
    }
}

QVariant ZenoNode::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (!m_bUIInited)
        return value;

    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        m_headerWidget->toggle(bSelected);
        m_bodyWidget->toggle(bSelected);

        ZenoSubGraphScene *pScene = qobject_cast<ZenoSubGraphScene *>(scene());
        ZASSERT_EXIT(pScene, value);
        const QString& name = m_index.data(ROLE_NODE_NAME).toString();
        pScene->collectNodeSelChanged(name, bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        m_bMoving = true;
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
        bool isSnapGrid = ZenoSettingsManager::GetInstance().getValue(zsSnapGrid).toBool();
        if (pScene) {
            if (isSnapGrid)
            {
                QPointF pos = value.toPointF();
                int x = pos.x(), y = pos.y();
                x = x - x % SCENE_GRID_SIZE;
                y = y - y % SCENE_GRID_SIZE;
                return QPointF(x, y);
            }
        }
        
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        m_bMoving = true;
        m_lastMoving = value.toPointF();
        emit inSocketPosChanged();
        emit outSocketPosChanged();
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
        if (pScene) {
            pScene->onNodePositionChanged(this);
        }
    }
    else if (change == ItemScenePositionHasChanged)
    {
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    }
    else if (change == ItemZValueHasChanged)
    {
        int type = m_index.data(ROLE_NODETYPE).toInt();
        if (type == zeno::Node_Group && zValue() != ZVALUE_BLACKBOARD)
        {
            setZValue(ZVALUE_BLACKBOARD);
        }
    }
    return value;
}

void ZenoNode::resizeEvent(QGraphicsSceneResizeEvent* event)
{
    _base::resizeEvent(event);
}

void ZenoNode::hoverEnterEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverEnterEvent(event);
}

void ZenoNode::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverMoveEvent(event);
}

void ZenoNode::hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
{
    _base::hoverLeaveEvent(event);
}

void ZenoNode::onOptionsBtnToggled(STATUS_BTN btn, bool toggled)
{
    zeno::NodeStatus options = (zeno::NodeStatus)m_index.data(ROLE_NODE_STATUS).toInt();
    int oldOpts = options;

    if (btn == STATUS_MUTE)
    {
        //TODO:
    }
    else if (btn == STATUS_VIEW)
    {
        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
        GraphModel* pGraphM = qobject_cast<GraphModel*>(pModel);
        ZASSERT_EXIT(pGraphM);
        pGraphM->setView(m_index, toggled);
    }
}

void ZenoNode::onCollaspeBtnClicked()
{
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    UiHelper::qIndexSetData(m_index, !bCollasped, ROLE_COLLASPED);
}

void ZenoNode::onCollaspeUpdated(bool collasped)
{
    m_mainHeaderBg->setVisible(collasped);
    m_headerWidget->setVisible(!collasped);
    m_bodyWidget->setVisible(!collasped);
    m_statusMarker->setVisible(!collasped);
    if (collasped) {
        m_topInputSockets->show();
        m_bottomOutputSockets->show();
    }
    else {
        m_topInputSockets->hide();
        m_bottomOutputSockets->hide();
    }
    updateWhole();
    update();
}

void ZenoNode::onViewUpdated(bool bView)
{
    if (m_pStatusWidgets)
    {
        m_pStatusWidgets->blockSignals(true);
        m_pStatusWidgets->setView(bView);
        m_pStatusWidgets->blockSignals(false);
    }
}

void ZenoNode::onOptionsUpdated(int options)
{
    if (m_pStatusWidgets) 
    {
        m_pStatusWidgets->blockSignals(true);
        m_pStatusWidgets->setOptions(options);
        m_pStatusWidgets->blockSignals(false);
    }
}

void ZenoNode::updateWhole()
{
    ZGraphicsLayout::updateHierarchy(this);
    emit inSocketPosChanged();
    emit outSocketPosChanged();
}

void ZenoNode::onUpdateFrame(QGraphicsItem* pContrl, int nFrame, QVariant val)
{
    QVariant newVal = val;
    if (!AppHelper::getCurveValue(newVal))
        return;
    //vec
    if (QGraphicsProxyWidget* pWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pContrl))
    {
        if (ZVecEditorItem* pEditor = qobject_cast<ZVecEditorItem*>(pWidget))
        {
            pEditor->setVec(newVal);
            QVector<QString> properties = AppHelper::getKeyFrameProperty(val);
            pEditor->updateProperties(properties);
        }
    }
    else if (QGraphicsTextItem* pTextItem = qgraphicsitem_cast<QGraphicsTextItem*>(pContrl))
    {
        pTextItem->setPlainText(UiHelper::variantToString(newVal));
        QVector<QString> properties = AppHelper::getKeyFrameProperty(val);
        pTextItem->setProperty(g_setKey, properties.first());
    }
}

void ZenoNode::onPasteSocketRefSlot(QModelIndex toIndex) 
{
    //Depreacated
}

void ZenoNode::onCustomNameChanged()
{
    m_NameItem->setTextInteractionFlags(Qt::NoTextInteraction);
    QString newText = m_NameItem->toPlainText();
    QString oldText = m_index.data(ROLE_NODE_NAME).toString();
    if (newText == oldText)
        return;

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    GraphModel* pGraphM = qobject_cast<GraphModel*>(pModel);
    ZASSERT_EXIT(pGraphM);

    newText = pGraphM->updateNodeName(m_index, newText);
    if (m_NameItem)
    {
        m_NameItem->setText(newText);
        m_NameItem->setDefaultTextColor(QColor(255, 255, 255));
        ZGraphicsLayout::updateHierarchy(m_NameItem);
    }
}