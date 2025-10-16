#include "zenonodenew.h"
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
#include "model/assetsmodel.h"
#include <zeno/core/typeinfo.h>


NodeNameItem::NodeNameItem(const QString& name, QGraphicsItem* parent)
    : ZGraphicsTextItem(name, QFont(), QColor("#FFFFFF"), parent)
{
    QFont font2;
    font2.setPointSize(12);
    font2.setWeight(QFont::Normal);
    setFont(font2);
    setData(GVKEY_SIZEPOLICY, QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred));
    //setTextInteractionFlags(Qt::TextEditable);
    //setAcceptHoverEvents(true);
}

void NodeNameItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    //override
    //ZGraphicsTextItem::mousePressEvent(event);
}

void NodeNameItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
    //override
    //ZGraphicsTextItem::mouseMoveEvent(event);
}

void NodeNameItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    ZGraphicsTextItem::mouseReleaseEvent(event);
}

void NodeNameItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
    setTextInteractionFlags(Qt::TextEditorInteraction);
    ZGraphicsTextItem::mouseDoubleClickEvent(event);
    setFocus(Qt::MouseFocusReason);
}

void NodeNameItem::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
        return;
    ZGraphicsTextItem::keyPressEvent(event);
}

void NodeNameItem::focusOutEvent(QFocusEvent* event) {
    ZGraphicsTextItem::focusOutEvent(event);
    setTextInteractionFlags(Qt::NoTextInteraction);
}


ZenoNodeNew::ZenoNodeNew(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(params, parent)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_inputObjSockets(nullptr)
    , m_outputObjSockets(nullptr)
    , m_nameEditor(nullptr)
    , m_nodeStatus(QmlNodeRunStatus::DirtyReadyToRun)
    , m_bodyLayout(nullptr)
    , m_inputsLayout(nullptr)
    , m_outputsLayout(nullptr)
    , m_pStatusWidgets1(nullptr)
    , m_pStatusWidgets2(nullptr)
    , m_NameItemTip(nullptr)
    , m_statusMarker(nullptr)
    , m_errorTip(nullptr)
    , m_frameNodeMark(nullptr)
    , m_nameItem(nullptr)
    , m_dirtyMarker(nullptr)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZenoNodeNew::~ZenoNodeNew()
{
}

void ZenoNodeNew::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    _base::paint(painter, option, widget);
    _drawShadow(painter);
}

QRectF ZenoNodeNew::boundingRect() const
{
    QRectF rect = _base::boundingRect();
    qreal topMargin = 6;
    qreal bottomMargin = 6;
    if (m_inputObjSockets)
        topMargin += m_inputObjSockets->geometry().height();
    if (m_outputObjSockets)
        bottomMargin += m_outputObjSockets->geometry().height();
    rect = rect - QMargins(0, topMargin, 0, bottomMargin);
    return rect;
}

void ZenoNodeNew::_drawShadow(QPainter* painter)
{
    QRectF rc = boundingRect();
    qreal offset = ZenoStyle::dpiScaled(4);
    if (m_nodeStatus == zeno::Node_RunError)
    {
        rc.adjust(2, 2, -2, -2);
    }
    else
    {
        offset += 16;
        rc.adjust(offset, offset, -2, -2);
    }

    QColor color= m_nodeStatus == zeno::Node_RunError ? QColor(192, 36, 36) : QColor(0, 0, 0);
    bool bCollasped = m_index.data(QtRole::ROLE_COLLASPED).toBool();

    int radius = 8;
    for (int i = 0; i < 16; i++)
    {
        QPainterPath path;
        rc.adjust(-1, -1, 1, 1);
        if (!bCollasped)
        {
            path = UiHelper::getRoundPath(rc, 0, radius, 0, 0, true);
        }
        else
        {
            path = UiHelper::getRoundPath(rc, radius, radius, radius, radius, true);
        }
        radius += 1;
        color.setAlpha(200 - qSqrt(i) * 50);
        QPen pen;
        pen.setJoinStyle(Qt::MiterJoin);
        pen.setWidth(1);
        pen.setColor(color);
        painter->setPen(pen);
        painter->drawPath(path);
    }
}


void ZenoNodeNew::initLayout()
{
    //ZASSERT_EXIT(m_index().isValid());

    //const QStringList& path = m_index.data(QtRole::ROLE_OBJPATH).toStringList();
    //m_dbgName = path.join("/");

    m_inputObjSockets = initVerticalSockets(true);
    m_headerWidget = initHeaderWidget();
    m_bodyWidget = initBodyWidget();
    m_outputObjSockets = initVerticalSockets(false);

    ZGraphicsLayout* mainLayout = new ZGraphicsLayout(false);
    mainLayout->setDebugName("mainLayout");
    mainLayout->addLayout(m_inputObjSockets);
    mainLayout->addSpacing(6);
    mainLayout->addItem(m_headerWidget);
    mainLayout->addItem(m_bodyWidget);
    mainLayout->addSpacing(1);
    mainLayout->addItem(m_dirtyMarker, Qt::AlignHCenter);
    mainLayout->addSpacing(3);
    mainLayout->addLayout(m_outputObjSockets);

    mainLayout->setSpacing(0);
    setLayout(mainLayout);
    setColors(false, QColor(0, 0, 0, 0));

    //更新一下脏位显示
    auto status = m_index.data(QtRole::ROLE_NODE_RUN_STATE).value<QmlNodeRunStatus::Value>();
    markNodeStatus(status);

    updateWhole();

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    connect(paramsM, &ParamsModel::enabledVisibleChanged, this, &ZenoNodeNew::onUpdateParamsVisbleEnabled);
}

void ZenoNodeNew::onUpdateParamsVisbleEnabled()
{
    for (ZenoSocketItem* pSocket : getObjSocketItems(true)) {
        QModelIndex paramIdx = pSocket->paramIndex();
        bool bEnable = paramIdx.data(QtRole::ROLE_PARAM_ENABLE).value<bool>();
        bool bVisible = paramIdx.data(QtRole::ROLE_PARAM_VISIBLE).value<bool>();
        pSocket->setVisible(bVisible);
        pSocket->setEnabled(bEnable);
    }
    for (ZenoSocketItem* pSocket : getObjSocketItems(false)) {
        QModelIndex paramIdx = pSocket->paramIndex();
        bool bEnable = paramIdx.data(QtRole::ROLE_PARAM_ENABLE).value<bool>();
        bool bVisible = paramIdx.data(QtRole::ROLE_PARAM_VISIBLE).value<bool>();
        pSocket->setVisible(bVisible);
        pSocket->setEnabled(bEnable);
    }
    updateWhole();
}

ZGraphicsLayout* ZenoNodeNew::initVerticalSockets(bool bInput)
{
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM, nullptr);

    ZGraphicsLayout* pSocketLayout = new ZGraphicsLayout(true);
    pSocketLayout->addSpacing(-1);
    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        if (paramIdx.data(QtRole::ROLE_ISINPUT).toBool() != bInput)
            continue;

        auto group = paramIdx.data(QtRole::ROLE_PARAM_GROUP).toInt();
        if (group != zeno::Role_InputObject && group != zeno::Role_OutputObject)
            continue;

        addOnlySocketToLayout(pSocketLayout, paramIdx);
    }
    pSocketLayout->addSpacing(-1);
    return pSocketLayout;
}

void ZenoNodeNew::setVisibleForParams(bool bVisible) {
    m_bodyWidget->setVisible(bVisible);

    //还要调整header ui块下面的corner radius.
    if (bVisible)
        m_headerWidget->setRadius(10, 10, 0, 0);
    else
        m_headerWidget->setRadius(10, 10, 10, 10);
    m_pStatusWidgets1->updateRightButtomRadius(!bVisible);
    m_pStatusWidgets2->updateRightButtomRadius(!bVisible);
    updateWhole();
}

void ZenoNodeNew::updateNodeNameByEditor() {
    ZGraphicsTextItem* textEditor = qobject_cast<ZGraphicsTextItem*>(sender());
    ZASSERT_EXIT(textEditor);
    QString newVal = textEditor->toPlainText();
    QString oldName = m_index.data(QtRole::ROLE_NODE_NAME).toString();
    if (newVal == oldName)
        return;
    if (GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_index.data(QtRole::ROLE_GRAPH)))
    {
        QString name = pModel->updateNodeName(m_index, newVal);
        if (name != newVal)
        {
            QMessageBox::warning(nullptr, tr("Rename warring"), tr("The name %1 is existed").arg(newVal));
            textEditor->setText(name);
        }
    }
    if (textEditor == m_nameItem)
        ZGraphicsLayout::updateHierarchy(textEditor);
}

ZLayoutBackground* ZenoNodeNew::initHeaderWidget()
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground;
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(10, 10, 10, 10);
    qreal bdrWidth = 0;// ZenoStyle::dpiScaled(headerBg.border_witdh);
    headerWidget->setBorder(bdrWidth, headerBg.clr_border);

    ZASSERT_EXIT(m_index.isValid(), nullptr);

    zeno::NodeType type = static_cast<zeno::NodeType>(m_index.data(QtRole::ROLE_NODETYPE).toInt());
    const QVariantMap& uistyle = m_index.data(QtRole::ROLE_NODE_UISTYLE).toMap();
    QString iconResPath = uistyle["icon"].toString();
    QString background = uistyle["background"].toString();

    QColor clrBgFrom, clrBgTo;
    if (type == zeno::NoVersionNode) {
        clrBgFrom = clrBgTo = QColor(83, 83, 85);
    }
    else if (type == zeno::Node_SubgraphNode || type == zeno::Node_AssetInstance || 
        type == zeno::Node_AssetReference) {
        clrBgFrom = QColor("#1A5447");
        clrBgTo = QColor("#289880");
    }
    else if (!background.isEmpty()) {
        clrBgFrom = clrBgTo = QColor(background);
    }
    else {
        clrBgFrom = QColor("#0277D1");
        clrBgTo = QColor("#0277D1");
    }

    //headerWidget->setColors(headerBg.bAcceptHovers, clrHeaderBg, clrHeaderBg, clrHeaderBg);
    headerWidget->setLinearGradient(clrBgFrom, clrBgTo);

    //headerWidget->setBorder(ZenoStyle::dpiScaled(headerBg.border_witdh), headerBg.clr_border);

    const QString& nodeCls = m_index.data(QtRole::ROLE_CLASS_NAME).toString();
    const QString& name = m_index.data(QtRole::ROLE_NODE_NAME).toString();
    //const QString& iconResPath = m_index.data(QtRole::ROLE_NODE_DISPLAY_ICON).toString();

    const QString& category = m_index.data(QtRole::ROLE_NODE_CATEGORY).toString();

    QFont font2 = QApplication::font();
    font2.setPointSize(12);
    font2.setWeight(QFont::Normal);

    m_nameItem = new NodeNameItem(name);
    m_nameItem->installEventFilter(this);
    //clsItem->setDefaultTextColor(QColor("#FFFFFF"));

    //qreal margin = ZenoStyle::dpiScaled(10);
    //pNameLayout->setContentsMargin(margin, margin, margin, margin);

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("Header HLayout");
    //pHLayout->addSpacing(ZenoStyle::dpiScaled(16.));

    //icons

    //ZGraphicsLayout* pNameLayout = new ZGraphicsLayout(true);

    connect(m_nameItem, &ZGraphicsTextItem::editingFinished, this, &ZenoNodeNew::updateNodeNameByEditor);
    ////TODO: 参照houdini，当名字与类名不重合时，就另外显示。
    //m_NameItem->hide();

    //pNameLayout->addSpacing(-1);
    const qreal W_status = ZenoStyle::dpiScaled(22.);
    const qreal H_status = ZenoStyle::dpiScaled(50.);
    const qreal radius = ZenoStyle::dpiScaled(9.);


    //要检查是否有可见基础类型的参数，如有，会调整header的ui
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    bool bBodyVisible = paramsM->hasVisiblePrimParam();

    RoundRectInfo buttonShapeInfo;
    buttonShapeInfo.W = ZenoStyle::dpiScaled(22.);
    buttonShapeInfo.H = ZenoStyle::dpiScaled(50.);
    buttonShapeInfo.ltradius = buttonShapeInfo.rtradius = ZenoStyle::dpiScaled(9.);
    //根据是否有visible的socket显示来决定
    buttonShapeInfo.lbradius = buttonShapeInfo.rbradius = bBodyVisible ? 0 : ZenoStyle::dpiScaled(9.);

    bool bHasOptimStatus = false;
    if (type == zeno::Node_SubgraphNode || type == zeno::Node_AssetInstance) {
        bHasOptimStatus = true;
    }

    bool bView = m_index.data(QtRole::ROLE_NODE_ISVIEW).toBool();
    bool bypass = m_index.data(QtRole::ROLE_NODE_BYPASS).toBool();
    bool nocache = m_index.data(QtRole::ROLE_NODE_NOCACHE).toBool();
    bool clearsubnet = m_index.data(QtRole::ROLE_NODE_CLEARSUBNET).toBool();

    //pHLayout->addLayout(pNameLayout);

    m_pStatusWidgets1 = new LeftStatusBtnGroup(type, buttonShapeInfo);
    m_pStatusWidgets1->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    m_pStatusWidgets1->setNoCache(nocache);
    m_pStatusWidgets1->setClearSubnet(clearsubnet);
    connect(m_pStatusWidgets1, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    m_pStatusWidgets2 = new RightStatusBtnGroup(buttonShapeInfo);
    m_pStatusWidgets2->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    m_pStatusWidgets2->setByPass(bypass);
    m_pStatusWidgets2->setView(bView);
    connect(m_pStatusWidgets2, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    pHLayout->addItem(m_pStatusWidgets1, Qt::AlignLeft);

    pHLayout->addSpacing(ZenoStyle::dpiScaled(16.));
    const QSizeF szIcon = ZenoStyle::dpiScaledSize(QSizeF(26, 26));

    if (zeno::getSession().is_frame_node(nodeCls.toStdString())) {
        m_frameNodeMark = new ZenoImageItem(":/icons/time-frame-mark.svg", "", "", QSizeF(20, 20), headerWidget);
        m_frameNodeMark->setPos(QPointF(-24, 28));
        m_frameNodeMark->setToolTip(tr("Every time the frame is changed, the node will be marked dirty"));
    }

    if (!iconResPath.isEmpty())
    {
        ImageElement elem;
        elem.image = elem.imageHovered = elem.imageOn = elem.imageOnHovered = iconResPath;
        auto node_icon = new ZenoImageItem(elem, szIcon);
        node_icon->setClickable(false);
        pHLayout->addItem(node_icon, Qt::AlignVCenter);

        m_nameEditor = new ZEditableTextItem(name, headerWidget);
        m_nameEditor->setDefaultTextColor(QColor("#CCCCCC"));
        m_nameEditor->setTextLengthAsBounding(true);
        m_nameEditor->setFont(font2);
        QRectF brNameEditor = m_nameEditor->boundingRect();
        qreal ww = brNameEditor.width() + ZenoStyle::dpiScaled(8);
        qreal nameEditor_height = 14;
        m_nameEditor->setPos(-ww, nameEditor_height);
        connect(m_nameEditor, &ZEditableTextItem::contentsChanged, this, [=]() {
            qreal ww = m_nameEditor->textLength() + ZenoStyle::dpiScaled(8);
            m_nameEditor->setPos(-ww, nameEditor_height);
        });
        if (m_frameNodeMark) {
            m_frameNodeMark->setPos(-24, nameEditor_height + brNameEditor.height());
        }
        connect(m_nameEditor, &ZGraphicsTextItem::editingFinished, this, &ZenoNodeNew::updateNodeNameByEditor);
    }
    else {
        //补充一些距离
        //pHLayout->addSpacing(szIcon.width());// +ZenoStyle::dpiScaled(20.));
        //pHLayout->addSpacing(100, QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed));
        pHLayout->addItem(m_nameItem, Qt::AlignVCenter);
    }
    pHLayout->addSpacing(ZenoStyle::dpiScaled(16.));

    pHLayout->addItem(m_pStatusWidgets2, Qt::AlignRight);

    m_dirtyMarker = new ZLayoutBackground;
    m_dirtyMarker->setColors(false, QColor(240, 215, 4));
    m_dirtyMarker->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    m_dirtyMarker->setGeometry(QRectF(0, 0, ZenoStyle::dpiScaled(50), ZenoStyle::dpiScaled(StatusButton::dirtyLayoutHeight)));

    ZGraphicsLayout* pVLayout = new ZGraphicsLayout(false);
    pVLayout->addLayout(pHLayout);

    headerWidget->setLayout(pVLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    if (const GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_index.data(QtRole::ROLE_GRAPH)))
    {
        m_pStatusWidgets1->setEnabled(!pModel->isLocked());
        m_pStatusWidgets2->setEnabled(!pModel->isLocked());
        connect(pModel, &GraphModel::lockStatusChanged, this, [=]() {
            m_pStatusWidgets1->setEnabled(!pModel->isLocked());
            m_pStatusWidgets2->setEnabled(!pModel->isLocked());
            for (auto layout : getSocketLayouts(true))
            {
                if (auto pControl = layout->control())
                    pControl->setEnabled(!pModel->isLocked());
            }
        });
    }



#if 0
    const NodeState& state = m_index.data(QtRole::ROLE_NODE_RUN_STATE).value<NodeState>();
    m_statusMarker = new QGraphicsPolygonItem(headerWidget);
    QPolygonF points;
    points.append(QPointF(0, 0));
    points.append(QPointF(15, 0));
    points.append(QPointF(0, 15));
    m_statusMarker->setPolygon(points);
    m_statusMarker->setPen(Qt::NoPen);
    m_statusMarker->setPos(QPointF(0, 0));
    markNodeStatus(state.runstatus);
#endif

    return headerWidget;
}

ZLayoutBackground* ZenoNodeNew::initBodyWidget()
{
    ZLayoutBackground* bodyWidget = new ZLayoutBackground(this);
    const auto& bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, QColor("#2A2A2A"));

    qreal bdrWidth = 0;// ZenoStyle::dpiScaled(bodyBg.border_witdh);
    //bodyWidget->setBorder(ZenoStyle::scaleWidth(2), QColor(18, 20, 22));

    m_bodyLayout = new ZGraphicsLayout(false);
    m_bodyLayout->setDebugName("Body Layout");
    qreal margin = ZenoStyle::dpiScaled(16);
    m_bodyLayout->setContentsMargin(margin, bdrWidth, 0, bdrWidth);

    ZASSERT_EXIT(m_index.isValid(), nullptr);
    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM, nullptr);

    connect(paramsM, &ParamsModel::rowsInserted, this, &ZenoNodeNew::onParamInserted);
    connect(paramsM, &ParamsModel::rowsAboutToBeRemoved, this, &ZenoNodeNew::onViewParamAboutToBeRemoved);
    connect(paramsM, &ParamsModel::dataChanged, this, &ZenoNodeNew::onParamDataChanged);
    connect(paramsM, &ParamsModel::rowsMoved, this, &ZenoNodeNew::onViewParamsMoved);
    bool ret = connect(paramsM, &ParamsModel::layoutChanged, this, &ZenoNodeNew::onLayoutChanged);

    if (auto pLayout = initCustomParamWidgets())
    {
        m_bodyLayout->addLayout(pLayout);
        m_bodyLayout->addSpacing(margin);
    }
    m_inputsLayout = initPrimSockets(paramsM, true);
    m_bodyLayout->addLayout(m_inputsLayout);

    m_outputsLayout = initPrimSockets(paramsM, false);
    m_bodyLayout->addLayout(m_outputsLayout);

    m_bodyLayout->addSpacing(13, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

    bodyWidget->setLayout(m_bodyLayout);

    bool bVisible = paramsM->hasVisiblePrimParam();
    bodyWidget->setVisible(bVisible);

    return bodyWidget;
}

void ZenoNodeNew::onRunStateChanged()
{
    auto status = m_index.data(QtRole::ROLE_NODE_RUN_STATE).value<QmlNodeRunStatus::Value>();
    markNodeStatus(status);
}

QVector<ZSocketLayout*> ZenoNodeNew::getSocketLayouts(bool bInput) const
{
    QVector<ZSocketLayout*> layouts;
    if (bInput) {
        if (!m_inputsLayout)
            return layouts;
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
        if (!m_outputsLayout)
            return layouts;
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

void ZenoNodeNew::addOnlySocketToLayout(ZGraphicsLayout* pSocketLayout, const QModelIndex& paramIdx)
{
    //外面有一个space
    if (pSocketLayout->count() == 1) {
        pSocketLayout->addSpacing(16);
    }

    QString name = paramIdx.data(QtRole::ROLE_PARAM_NAME).toString();
    QFontMetrics fontMetrics(font());
    qreal xmargin = 10;
    qreal ymargin = 0;
    QSizeF szSocket(fontMetrics.width(name) + xmargin, fontMetrics.height() + ymargin);
    ZenoSocketItem* socket = new ZenoObjSocketItem(paramIdx, ZenoStyle::dpiScaledSize(szSocket));
    socket->setBrush(QColor("#CCA44E"), QColor("#ee9922"));
    pSocketLayout->addItem(socket);
    pSocketLayout->addSpacing(ZenoStyle::dpiScaled(16));

    if (ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS))) {
        QObject::connect(paramsM, &QStandardItemModel::dataChanged, socket, &ZenoSocketItem::onCustomParamDataChanged);
    }

    QObject::connect(socket, &ZenoSocketItem::clicked, [=](bool bInput) {
        emit socketClicked(socket);
    });
}

void ZenoNodeNew::onLayoutChanged()
{
    m_inputsLayout->clear();
    m_inputObjSockets->clear();
    m_outputsLayout->clear();
    m_outputObjSockets->clear();

    m_inputObjSockets->addSpacing(-1);
    m_outputObjSockets->addSpacing(-1);

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);

    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        auto group = paramIdx.data(QtRole::ROLE_PARAM_GROUP).toInt();
        if (!paramIdx.data(QtRole::ROLE_ISINPUT).toBool())
            continue;

        if (group == zeno::Role_InputObject)
            addOnlySocketToLayout(m_inputObjSockets, paramIdx);
        else
            m_inputsLayout->addItem(addSocket(paramIdx, true));
    }

    for (int r = 0; r < paramsM->rowCount(); r++)
    {
        const QModelIndex& paramIdx = paramsM->index(r, 0);
        auto group = paramIdx.data(QtRole::ROLE_PARAM_GROUP).toInt();
        if (paramIdx.data(QtRole::ROLE_ISINPUT).toBool())
            continue;

        if (group == zeno::Role_OutputObject)
            addOnlySocketToLayout(m_outputObjSockets, paramIdx);
        else
            m_outputsLayout->addItem(addSocket(paramIdx, false));
    }

    m_inputObjSockets->addSpacing(-1);
    m_outputObjSockets->addSpacing(-1);

    bool bCollasped = m_index.data(QtRole::ROLE_COLLASPED).toBool();
    onCollaspeUpdated(bCollasped);

    updateWhole();
}

ZGraphicsLayout* ZenoNodeNew::initCustomParamWidgets()
{
    return nullptr;
}

void ZenoNodeNew::onNameUpdated(const QString& newName)
{
    ZASSERT_EXIT(m_nameItem);
    if (m_nameItem && newName != m_nameItem->toPlainText())
    {
        m_nameItem->setText(newName);
        ZGraphicsLayout::updateHierarchy(m_nameItem);
    }
}

ZSocketLayout* ZenoNodeNew::getSocketLayout(bool bInput, const QString& name) const
{
    QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
    for (int i = 0; i < layouts.size(); i++)
    {
        QModelIndex idx = layouts[i]->viewSocketIdx();
        QString sockName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
        if (sockName == name)
            return layouts[i];
    }
    return nullptr;
}

ZSocketLayout* ZenoNodeNew::getSocketLayout(bool bInput, int idx) const
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

bool ZenoNodeNew::removeSocketLayout(bool bInput, const QString& name)
{
    if (bInput)
    {
        QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
        for (int i = 0; i < layouts.size(); i++)
        {
            QModelIndex idx = layouts[i]->viewSocketIdx();
            QString sockName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
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
            QString sockName = idx.data(QtRole::ROLE_PARAM_NAME).toString();
            if (sockName == name)
            {
                m_outputsLayout->removeElement(i);
                return false;
            }
        }
    }
    return false;
}

void ZenoNodeNew::onParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (roles.isEmpty())
        return;

    if (!m_index.isValid())
        return;

    QModelIndex paramIdx = topLeft;

    for (int role : roles)
    {
        if (role != QtRole::ROLE_PARAM_NAME
            && role != QtRole::ROLE_PARAM_TOOLTIP
            && role != QtRole::ROLE_PARAM_SOCKET_VISIBLE
            && role != QtRole::ROLE_PARAM_GROUP)
            return;

        const bool bInput = paramIdx.data(QtRole::ROLE_ISINPUT).toBool();
        const QString& paramName = paramIdx.data(QtRole::ROLE_PARAM_NAME).toString();

        if (role == QtRole::ROLE_PARAM_NAME || role == QtRole::ROLE_PARAM_TOOLTIP || role == QtRole::ROLE_PARAM_SOCKET_VISIBLE)
        {
            QVector<ZSocketLayout*> layouts = getSocketLayouts(bInput);
            for (int i = 0; i < layouts.size(); i++)
            {
                ZSocketLayout* pSocketLayout = layouts[i];
                QModelIndex socketIdx = pSocketLayout->viewSocketIdx();
                if (socketIdx == paramIdx)
                {
                    if (role == QtRole::ROLE_PARAM_NAME)
                        pSocketLayout->updateSockName(paramName);   //only update name on control.
                    else if (role == QtRole::ROLE_PARAM_TOOLTIP)
                        pSocketLayout->updateSockNameToolTip(paramIdx.data(QtRole::ROLE_PARAM_TOOLTIP).toString());
                    else if (role == QtRole::ROLE_PARAM_SOCKET_VISIBLE)
                    {
                        auto bVisible = paramIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
                        pSocketLayout->setVisible(bVisible);
                        pSocketLayout->setSocketVisible(bVisible);
                        //layout上面的那个SocketBackground也得设为可见
                        if (SocketBackgroud* pBg = pSocketLayout->getSocketBg()) {
                            pBg->setVisible(bVisible);
                        }
                        emit inSocketPosChanged();
                    }
                    break;
                }
            }

            if (role == QtRole::ROLE_PARAM_SOCKET_VISIBLE) {
                bool bVisible = paramIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
                if (bVisible) {
                    setVisibleForParams(true);
                }
                else {
                    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
                    ZASSERT_EXIT(paramsM);
                    setVisibleForParams(paramsM->hasVisiblePrimParam());
                }
            }

            updateWhole();
            return;
        }
    }
}

void ZenoNodeNew::onParamInserted(const QModelIndex& parent, int first, int last)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);

    for (int r = first; r <= last; r++)
    {
        QModelIndex paramIdx = paramsM->index(r, 0, parent);
        bool bInput = paramIdx.data(QtRole::ROLE_ISINPUT).toBool();
        ZGraphicsLayout* pSocketsLayout = bInput ? m_inputsLayout : m_outputsLayout;
        pSocketsLayout->addItem(addSocket(paramIdx, bInput));
        updateWhole();
    }
}

void ZenoNodeNew::onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int destRow)
{
    ParamsModel* paramsM = qobject_cast<ParamsModel*>(sender());
    ZASSERT_EXIT(paramsM);

    //TODO: output case
    m_inputsLayout->moveItem(start, destRow);
    updateWhole();
}

void ZenoNodeNew::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (!parent.isValid())
    {
        //remove all component.
        m_inputsLayout->clear();
        m_outputsLayout->clear();
        //m_inSockets.clear();
        //m_outSockets.clear();
        return;
    }

    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;

    ParamsModel* paramsM = QVariantPtr<ParamsModel>::asPtr(m_index.data(QtRole::ROLE_PARAMS));
    ZASSERT_EXIT(paramsM);
    for (int r = first; r <= last; r++)
    {
        QModelIndex viewParamIdx = paramsM->index(r, 0, parent);
        const int paramCtrl = viewParamIdx.data(QtRole::ROLE_PARAM_CONTROL).toInt();
        bool bInput = viewParamIdx.data(QtRole::ROLE_ISINPUT).toBool();

        const QString& paramName = viewParamIdx.data(QtRole::ROLE_PARAM_NAME).toString();
        ZSocketLayout* pSocketLayout = getSocketLayout(bInput, paramName);
        removeSocketLayout(bInput, paramName);

        ZASSERT_EXIT(pSocketLayout);
        ZGraphicsLayout* pParentLayout = pSocketLayout->parentLayout();
        pParentLayout->removeLayout(pSocketLayout);
        updateWhole();
    }
}

ZGraphicsLayout* ZenoNodeNew::initPrimSockets(ParamsModel* pModel, const bool bInput)
{
    ZASSERT_EXIT(pModel, nullptr);

    ZGraphicsLayout* pSocketsLayout = new ZGraphicsLayout(false);
    pSocketsLayout->setSpacing(0);

    for (int r = 0; r < pModel->rowCount(); r++)
    {
        const QModelIndex& paramIdx = pModel->index(r, 0);
        if (paramIdx.data(QtRole::ROLE_ISINPUT).toBool() != bInput)
            continue;
        auto group = paramIdx.data(QtRole::ROLE_PARAM_GROUP).toInt();
        if (group != zeno::Role_InputPrimitive && group != zeno::Role_OutputPrimitive)
            continue;
        pSocketsLayout->addItem(addSocket(paramIdx, bInput));
    }
    return pSocketsLayout;
}

SocketBackgroud* ZenoNodeNew::addSocket(const QModelIndex& paramIdx, bool bInput)
{
    QPersistentModelIndex perSockIdx = paramIdx;

    CallbackForSocket cbSocket;
    cbSocket.cbOnSockClicked = [=](ZenoSocketItem* pSocketItem) {
        emit socketClicked(pSocketItem);
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

    const QString& sockName = paramIdx.data(QtRole::ROLE_PARAM_NAME).toString();
    const zeno::ParamType type = (zeno::ParamType)paramIdx.data(QtRole::ROLE_PARAM_TYPE).toLongLong();

    zeno::NodeType nodetype = static_cast<zeno::NodeType>(m_index.data(QtRole::ROLE_NODETYPE).toInt());
    bool bSocketEnable = true;
    if (SOCKPROP_LEGACY == paramIdx.data(QtRole::ROLE_PARAM_SOCKPROP) || nodetype == zeno::NoVersionNode)
    {
        bSocketEnable = false;
    }

    SocketBackgroud* pBackground = nullptr;
    ZSocketLayout* pMiniLayout = nullptr;
    if (type == gParamType_Dict || type == gParamType_List) {
        pBackground = new SocketBackgroud(bInput, true);
        pMiniLayout = new ZDictSocketLayout(paramIdx, bInput, pBackground);
    }
    else {
        pBackground = new SocketBackgroud(bInput, false);
        pMiniLayout = new ZSocketLayout(paramIdx, bInput, pBackground);
        qreal margin = ZenoStyle::dpiScaled(16);
        if (bInput)
            pMiniLayout->setContentsMargin(0, 0, 0, margin);
    }

    pMiniLayout->initUI(cbSocket);
    pMiniLayout->setDebugName(sockName);
    bool bVisible = paramIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
    pMiniLayout->setVisible(bVisible);
    pMiniLayout->setSocketVisible(bVisible);

    pBackground->setLayout(pMiniLayout);
    pBackground->setVisible(bVisible);
    return pBackground;
}


void ZenoNodeNew::onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded, const QString keyName)
{
    ZenoSocketItem* pSocket = getSocketItem(paramIdx, keyName);
    if (pSocket == nullptr)
        return;

    QModelIndex idx = pSocket->paramIndex();
    // the removal of links from socket is executed before the removal of link itself.
    PARAM_LINKS links = idx.data(QtRole::ROLE_LINKS).value<PARAM_LINKS>();
    ZenoSocketItem::SOCK_STATUS status = ZenoSocketItem::STATUS_UNKNOWN;
    if (bAdded) {
        status = ZenoSocketItem::STATUS_CONNECTED;
    } else {
        if (links.isEmpty())
            status = ZenoSocketItem::STATUS_NOCONN;
    }
    if (status != ZenoSocketItem::STATUS_UNKNOWN)
    {
        pSocket->setSockStatus(status);
        if (auto pItem = getObjSocketItem(paramIdx, bInput))
            pItem->setSockStatus(status);
    }

    if (bInput)
    {
        QString sockName = paramIdx.data(QtRole::ROLE_PARAM_NAME).toString();
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

void ZenoNodeNew::markNodeStatus(QmlNodeRunStatus::Value status)
{
    m_nodeStatus = status;
    QColor clrHeaderBg;
    if (m_nodeStatus == QmlNodeRunStatus::RunError)
    {
        qreal szIcon = ZenoStyle::dpiScaled(64);
        if (!m_errorTip) {
            m_errorTip = new ZenoImageItem(":/icons/node/error.svg", "", "",
                QSize(szIcon, szIcon), this);
        }
        m_errorTip->setPos(QPointF(-szIcon/2., -szIcon/2.));
        m_errorTip->setZValue(ZVALUE_ERRORTIP);
        m_errorTip->show();
    }
    else if (m_nodeStatus == QmlNodeRunStatus::DirtyReadyToRun ||
             m_nodeStatus == QmlNodeRunStatus::RunSucceed)
    {
        QColor clrMarker;
        if (m_nodeStatus != QmlNodeRunStatus::RunSucceed)
            clrMarker = QColor(240, 215, 4);
        else
            clrMarker = QColor(0, 0, 0, 0);
        ZASSERT_EXIT(m_dirtyMarker);
        m_dirtyMarker->setColors(false, clrMarker);
    }
    else {
        if (m_errorTip)
            m_errorTip->hide();

        if (m_statusMarker) {
            switch (m_nodeStatus)
            {
            case QmlNodeRunStatus::RunSucceed: {
                m_statusMarker->setBrush(QBrush(QColor("#319E36")));
                break;
            }
            case QmlNodeRunStatus::Pending: {
                m_statusMarker->setBrush(QBrush(QColor("#868686")));
                break;
            }
            case QmlNodeRunStatus::DirtyReadyToRun: {
                m_statusMarker->setBrush(QBrush(QColor("#EAED4B")));
                break;
            }
            case QmlNodeRunStatus::Running: {
                m_statusMarker->setBrush(QBrush(QColor("#02F8F8")));
                break;
            }
            }
            m_statusMarker->update();
        }
    }
    update();
}


QVector<ZenoSocketItem*> ZenoNodeNew::getSocketItems(bool bInput) const
{
    QVector<ZenoSocketItem*> sockets = getObjSocketItems(bInput);
    auto socks = getSocketLayouts(bInput);
    for (ZSocketLayout* sock : socks)
    {
        ZenoSocketItem* pSocketItem = sock->socketItem();
        sockets.append(pSocketItem);
    }

    return sockets;
}

QVector<ZenoSocketItem*> ZenoNodeNew::getObjSocketItems(bool bInput) const
{
    QVector<ZenoSocketItem*> sockets;
    ZGraphicsLayout* socketsLayout = bInput ? m_inputObjSockets : m_outputObjSockets;
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
    return sockets;
}

ZenoSocketItem* ZenoNodeNew::getObjSocketItem(const QModelIndex& sockIdx, bool bInput)
{
    ZGraphicsLayout* socketsLayout = bInput ? m_inputObjSockets : m_outputObjSockets;
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

ZenoSocketItem* ZenoNodeNew::getSocketItem(const QModelIndex& sockIdx, const QString keyName)
{
    const bool bInput = sockIdx.data(QtRole::ROLE_ISINPUT).toBool();
    if (ZenoSocketItem* pItam = getObjSocketItem(sockIdx, bInput))
    {
        return pItam;
    }
    else
    {
        if (ZenoSocketItem* pItem = getPrimSocketItem(sockIdx, bInput, keyName))
        {
            return pItem;
        }

        return nullptr;
    }
}

ZenoSocketItem* ZenoNodeNew::getPrimSocketItem(const QModelIndex& sockIdx, bool bInput, const QString keyName)
{
    for (ZSocketLayout* socklayout : getSocketLayouts(bInput))
    {
        if (ZenoSocketItem* pItem = socklayout->socketItemByIdx(sockIdx, keyName))
        {
            return pItem;
        }
    }
    return nullptr;
}

ZenoSocketItem* ZenoNodeNew::getNearestSocket(const QPointF& pos, bool bInput)
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

QModelIndex ZenoNodeNew::getSocketIndex(QGraphicsItem* uiitem, bool bSocketText) const
{
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

QPointF ZenoNodeNew::getSocketPos(const QModelIndex& sockIdx, const QString keyName)
{
    ZASSERT_EXIT(sockIdx.isValid(), QPointF());
    const bool bInput = sockIdx.data(QtRole::ROLE_ISINPUT).toBool();
    if (ZenoSocketItem* pSocket = getObjSocketItem(sockIdx, bInput))
    {
        if (bInput)
            return QPointF(pSocket->center().x(), pSocket->sceneBoundingRect().top());
        else
            return QPointF(pSocket->center().x(), pSocket->sceneBoundingRect().bottom());
    }
    bool bCollasped = m_index.data(QtRole::ROLE_COLLASPED).toBool();
    bool bVisible = sockIdx.data(QtRole::ROLE_PARAM_SOCKET_VISIBLE).toBool();
    if (bCollasped || !bVisible)
    {
        //zeno::log_warn("socket pos error");
        QRectF rc = m_headerWidget->sceneBoundingRect();
        if (bInput) {
            return QPointF(rc.left(), rc.center().y());
        }
        else {
            return QPointF(rc.right(), rc.center().y());
        }
    }
    else
    {
        if (ZenoSocketItem* pSocket = getPrimSocketItem(sockIdx, bInput, keyName))
        {
            return pSocket->center();
        }
    }
    zeno::log_warn("socket pos error");
    return QPointF(0, 0);
}


void ZenoNodeNew::onZoomed()
{
    //m_pStatusWidgets->onZoomed();
    qreal factor = 0.2;
    bool bVisible = true;
    if (editor_factor < factor) {
        bVisible = false;
    }
    

    if (editor_factor < 0.1 || editor_factor > 0.25) 
    {
        if (m_NameItemTip) 
        {
            delete m_NameItemTip;
            m_NameItemTip = nullptr;
        }
    }
    else if (m_NameItemTip == nullptr) 
    {

        const QString& nodeCls = m_index.data(QtRole::ROLE_NODE_NAME).toString();
        m_NameItemTip = new ZSimpleTextItem(nodeCls, this);

        QFont font2 = QApplication::font();
        font2.setPointSize(9);
        font2.setWeight(QFont::ExtraLight);

        m_NameItemTip->setBrush(QColor("#CCCCCC"));
        m_NameItemTip->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        m_NameItemTip->setFont(font2);
        m_NameItemTip->show();

    }
    if (m_NameItemTip) 
    {
        QString name = m_index.data(QtRole::ROLE_NODE_NAME).toString();
        if (m_NameItemTip->text() != name)
            m_NameItemTip->setText(name);
        m_NameItemTip->setPos(QPointF(m_headerWidget->pos().x(), -ZenoStyle::scaleWidth(36)));
    }
}

bool ZenoNodeNew::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
    if (event->type() == QEvent::GraphicsSceneMousePress) {
        int j;
        j = 0;
    }
    else if (event->type() == QEvent::GraphicsSceneMouseMove) {
        int j;
        j = 0;
    }
    return _base::sceneEventFilter(watched, event);
}

bool ZenoNodeNew::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == m_nameEditor)
    {
        if ((event->type() == QEvent::InputMethod || event->type() == QEvent::KeyPress) && m_nameEditor->textInteractionFlags() == Qt::TextEditable)
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
                QString name = m_index.data(QtRole::ROLE_CLASS_NAME).toString();
                QColor color = QColor(255, 255, 255);
                QColor textColor = m_nameEditor->defaultTextColor();
                if (textColor != color)
                {
                    m_nameEditor->setDefaultTextColor(color);
                }

                if (m_nameEditor->toPlainText() == name)
                {
                    m_nameEditor->setText("");
                }

                if (m_nameEditor->textInteractionFlags() != Qt::TextEditorInteraction)
                {
                    m_nameEditor->setTextInteractionFlags(Qt::TextEditorInteraction);
                }
            }
        }
        else if (event->type() == QEvent::KeyRelease && m_nameEditor->textInteractionFlags() == Qt::TextEditorInteraction)
        {
            QString text = m_nameEditor->toPlainText();
            if (text.isEmpty())
            {
                QString name = m_index.data(QtRole::ROLE_CLASS_NAME).toString();
                m_nameEditor->setText(name);
                m_nameEditor->setTextInteractionFlags(Qt::TextEditable);
                m_nameEditor->setDefaultTextColor(QColor(255, 255, 255, 40));
            }
        }
    }
    else if (obj == m_nameItem) {
        QEvent::Type type = event->type();
        switch (type)
        {
            case QEvent::GraphicsSceneMousePress: {
                QGraphicsSceneMouseEvent* mouseEvent = static_cast<QGraphicsSceneMouseEvent*>(event);
                _cache_name_move = mouseEvent->scenePos();
                _cache_origin_pos = _cache_name_move;
                if (!isSelected()) {
                    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
                    ZASSERT_EXIT(pScene, false);
                    pScene->select({ m_index });
                }
                break;
            }
            case QEvent::GraphicsSceneMouseRelease: {
                QGraphicsSceneMouseEvent* mouseEvent = static_cast<QGraphicsSceneMouseEvent*>(event);
                QPointF mousePos = mouseEvent->scenePos();
                ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
                ZASSERT_EXIT(pScene, false);
                if (isSelected() &&
                    pScene->selectedItems().size() > 1 &&
                    qAbs(mouseEvent->scenePos().x() - _cache_origin_pos.x()) < 3 &&
                    qAbs(mouseEvent->scenePos().y() - _cache_origin_pos.y()) < 3) {
                    pScene->select({ m_index });
                }
                mouseReleaseEvent(mouseEvent);
                break;
            }
            case QEvent::GraphicsSceneMouseMove: {
                QGraphicsSceneMouseEvent* mouseEvent = static_cast<QGraphicsSceneMouseEvent*>(event);
                QPointF mousePos = mouseEvent->scenePos();
                QPointF offset = mousePos - _cache_name_move;
                auto selectedItems = scene()->selectedItems();
                for (int i = 0; i < selectedItems.size(); i++) {
                    auto cur = selectedItems.at(i);
                    if (ZenoNodeBase* base = qgraphicsitem_cast<ZenoNodeBase*>(cur)) {
                        cur->setPos(cur->scenePos() + offset);
                    }
                }
                _cache_name_move = mousePos;
                break;
            }
        }
    }
    return _base::eventFilter(obj, event);
}


void ZenoNodeNew::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
    QList<QGraphicsItem*> items = scene()->items(event->scenePos());
    if (items.contains(m_nameEditor))
    {
        if (const GraphModel* pModel = QVariantPtr<GraphModel>::asPtr(m_index.data(QtRole::ROLE_GRAPH)))
        {
            if (pModel->isLocked())
                return;
        }
        QString name = m_index.data(QtRole::ROLE_CLASS_NAME).toString();
        if (name == m_nameEditor->toPlainText())
        {
            m_nameEditor->setTextInteractionFlags(Qt::TextEditable);
            m_nameEditor->setDefaultTextColor(QColor(255, 255, 255, 40));
        }
        else
        {
            m_nameEditor->setTextInteractionFlags(Qt::TextEditorInteraction);
            m_nameEditor->setDefaultTextColor(QColor(255, 255, 255));
        }
        m_nameEditor->setFocus();
    }
    else if (items.contains(m_headerWidget) || items.contains(m_bodyWidget))
    {
        const QModelIndex& nodeIdx = index();
        zeno::NodeType type = (zeno::NodeType)nodeIdx.data(QtRole::ROLE_NODETYPE).toInt();
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
                QString assetName = nodeIdx.data(QtRole::ROLE_CLASS_NAME).toString();
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


QVariant ZenoNodeNew::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (!m_bUIInited)
        return value;
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        m_headerWidget->toggle(bSelected);
        if (m_bodyWidget)
            m_bodyWidget->toggle(bSelected);
    }
    _base::itemChange(change, value);
    return value;
}


void ZenoNodeNew::onOptionsBtnToggled(STATUS_BTN btn, bool toggled)
{
    zeno::NodeStatus options = (zeno::NodeStatus)m_index.data(QtRole::ROLE_NODE_STATUS).toInt();
    int oldOpts = options;

    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    GraphModel* pGraphM = qobject_cast<GraphModel*>(pModel);
    ZASSERT_EXIT(pGraphM);

    if (btn == STATUS_BYPASS) {
        pGraphM->setBypass(m_index, toggled);
    }
    else if (btn == STATUS_VIEW) {
        pGraphM->setView(m_index, toggled);
    }
    else if (btn == STATUS_CLEARSUBNET) {
        pGraphM->setClearSubnet(m_index, toggled);
    }
    else if (btn == STATUS_NOCACHE) {
        pGraphM->setNocache(m_index, toggled);
    }
}

void ZenoNodeNew::onCollaspeBtnClicked()
{
#if 0
    bool bCollasped = m_index.data(QtRole::ROLE_COLLASPED).toBool();
    QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    if (GraphModel* model = qobject_cast<GraphModel*>(pModel))
    {
        model->setModelData(m_index, !bCollasped, QtRole::ROLE_COLLASPED);
    }
#endif
}

void ZenoNodeNew::onCollaspeUpdated(bool collasped)
{
#if 0
    if (m_bodyWidget)
        m_bodyWidget->setVisible(!collasped);
    updateWhole();
    update();
#endif
}

void ZenoNodeNew::onViewUpdated(bool bView)
{
    ZASSERT_EXIT(m_pStatusWidgets2);
    m_pStatusWidgets2->blockSignals(true);
    m_pStatusWidgets2->setView(bView);
    m_pStatusWidgets2->blockSignals(false);
}

void ZenoNodeNew::onByPassUpdated(bool bypass) {
    ZASSERT_EXIT(m_pStatusWidgets2);
    m_pStatusWidgets2->blockSignals(true);
    m_pStatusWidgets2->setByPass(bypass);
    m_pStatusWidgets2->blockSignals(false);
}

void ZenoNodeNew::onNoCachedUpdated(bool nocache) {
    ZASSERT_EXIT(m_pStatusWidgets1);
    m_pStatusWidgets1->blockSignals(true);
    m_pStatusWidgets1->setNoCache(nocache);
    m_pStatusWidgets1->blockSignals(false);
}

void ZenoNodeNew::onClearSubnetUpdated(bool clearSubnet) {
    ZASSERT_EXIT(m_pStatusWidgets1);
    m_pStatusWidgets1->blockSignals(true);
    m_pStatusWidgets1->setClearSubnet(clearSubnet);
    m_pStatusWidgets1->blockSignals(false);
}

void ZenoNodeNew::onOptionsUpdated(int options)
{
    //DEPRECATED
    assert(false);
}
