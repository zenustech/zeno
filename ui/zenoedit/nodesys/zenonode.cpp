#include "zenonode.h"
#include "zenosubgraphscene.h"
#include <zenomodel/include/modelrole.h>
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenoparamnameitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenomodel/include/uihelper.h>
#include <zenomodel/include/igraphsmodel.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/scope_exit.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zveceditor.h>
#include "variantptr.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <zenomodel/include/graphsmanagment.h>
#include "../nodesview/zenographseditor.h"
#include "util/log.h"
#include "zenosubgraphview.h"
#include <zenoui/comctrl/dialog/zenoheatmapeditor.h>
#include <zenoui/comctrl/gv/zitemfactory.h>
#include "zvalidator.h"
#include "zenonewmenu.h"
#include "util/apphelper.h"
#include "viewport/viewportwidget.h"
#include "viewport/displaywidget.h"
#include <zenoui/comctrl/gv/zgraphicstextitem.h>
#include <zenoui/comctrl/gv/zenogvhelper.h>
#include <zenomodel/include/iparammodel.h>
#include <zenomodel/include/viewparammodel.h>
#include "iotags.h"
#include "groupnode.h"
#include "dialog/zeditparamlayoutdlg.h"
#include "settings/zenosettingsmanager.h"
#include <zenomodel/include/command.h>
#include <zenomodel/include/nodeparammodel.h>


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_border(new QGraphicsRectItem)
    , m_NameItem(nullptr)
    , m_pCategoryItem(nullptr)
    , m_bError(false)
    , m_bEnableSnap(false)
    , m_bMoving(false)
    , m_bodyLayout(nullptr)
    , m_bUIInited(false)
    , m_inputsLayout(nullptr)
    , m_paramsLayout(nullptr)
    , m_outputsLayout(nullptr)
    , m_groupNode(nullptr)
    , m_pStatusWidgets(nullptr)
    , m_bVisible(true)
    , m_NameItemTip(nullptr)
    , m_dirtyMarker(nullptr)
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
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());
    if (type == NORMAL_NODE || type == HEATMAP_NODE || type == SUBINPUT_NODE || type == SUBOUTPUT_NODE) {
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
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    m_headerWidget = initHeaderWidget(pGraphsModel);
    m_bodyWidget = initBodyWidget(pScene);

    ZGraphicsLayout* mainLayout = new ZGraphicsLayout(false);
    mainLayout->setDebugName("mainLayout");
    mainLayout->addItem(m_headerWidget);
    mainLayout->addItem(m_bodyWidget);

    mainLayout->setSpacing(0);
    setLayout(mainLayout);

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_OBJID).toString();
    setPos(pos);

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped)
        onCollaspeUpdated(true);

    // setPos will send geometry, but it's not supposed to happend during initialization.
    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    updateWhole();

    if (type == BLACKBOARD_NODE || type == GROUP_NODE) {
        setZValue(ZVALUE_BLACKBOARD);
    } else {
        //set color for normal node(background)
        setColors(false, QColor("#000000"), QColor("#000000"), QColor("#000000"));
    }

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();

    m_bUIInited = true;
    onZoomed();
}

ZLayoutBackground* ZenoNode::initHeaderWidget(IGraphsModel* pGraphsModel)
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground;
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
    qreal bdrWidth = ZenoStyle::dpiScaled(headerBg.border_witdh);
    headerWidget->setBorder(bdrWidth, headerBg.clr_border);

    ZASSERT_EXIT(m_index.isValid(), nullptr);

    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

    QColor clrHeaderBg;
    if (type == NO_VERSION_NODE)
        clrHeaderBg = QColor(83, 83, 85);
    else if (pGraphsModel->IsSubGraphNode(m_index))
        clrHeaderBg = QColor("#1D5F51");
    else
        clrHeaderBg = headerBg.clr_normal;

    headerWidget->setColors(headerBg.bAcceptHovers, clrHeaderBg, clrHeaderBg, clrHeaderBg);
    headerWidget->setBorder(ZenoStyle::dpiScaled(headerBg.border_witdh), headerBg.clr_border);

    const QString& name = m_index.data(ROLE_OBJNAME).toString();
    NODE_DESC desc;
    pGraphsModel->getDescriptor(name, desc);
    QString category;
    if (!desc.categories.isEmpty())
        category = desc.categories[0];

    ZGraphicsLayout* pNameLayout = new ZGraphicsLayout(false);
    qreal margin = ZenoStyle::dpiScaled(10);
    pNameLayout->setContentsMargin(margin, margin, margin, margin);

    QFont font2 = QApplication::font();
    font2.setPointSize(16);
    font2.setWeight(QFont::DemiBold);
    QString custName = m_index.data(ROLE_CUSTOM_OBJNAME).toString();
    m_NameItem = new ZGraphicsTextItem(custName.isEmpty() ? name : custName, font2, QColor("#FFFFFF"), this);
    m_NameItem->installEventFilter(this);
    connect(m_NameItem, &ZGraphicsTextItem::editingFinished, this, &ZenoNode::onCustomNameChanged);

    pNameLayout->addItem(m_NameItem);
    if (!category.isEmpty())
    {
        m_pCategoryItem = new ZSimpleTextItem(custName.isEmpty() ? category : category + ":" + name);
        m_pCategoryItem->setBrush(QColor("#AB6E40"));
        QFont font = QApplication::font();
        m_pCategoryItem->setFont(font);
        m_pCategoryItem->updateBoundingRect();
        m_pCategoryItem->setAcceptHoverEvents(false);
        pNameLayout->addItem(m_pCategoryItem);
    }

    m_pStatusWidgets = new ZenoMinStatusBtnItem(m_renderParams.status);
    int options = m_index.data(ROLE_OPTIONS).toInt();
    m_pStatusWidgets->setOptions(options);
    connect(m_pStatusWidgets, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->setDebugName("Header HLayout");
    pHLayout->addSpacing(10);
    pHLayout->addLayout(pNameLayout);
    pHLayout->addSpacing(100, QSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed));
    pHLayout->addItem(m_pStatusWidgets);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    return headerWidget;
}

ZLayoutBackground* ZenoNode::initBodyWidget(ZenoSubGraphScene* pScene)
{
    ZLayoutBackground* bodyWidget = new ZLayoutBackground(this);
    const auto& bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal);

    qreal bdrWidth = ZenoStyle::dpiScaled(bodyBg.border_witdh);
    bodyWidget->setBorder(ZenoStyle::scaleWidth(2), QColor(18, 20, 22));

    m_bodyLayout = new ZGraphicsLayout(false);
    m_bodyLayout->setDebugName("Body Layout");
    qreal margin = ZenoStyle::dpiScaled(16);
    m_bodyLayout->setContentsMargin(margin, bdrWidth, 0, bdrWidth);

    ZASSERT_EXIT(m_index.isValid(), nullptr);
    NodeParamModel* viewParams = QVariantPtr<NodeParamModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams, nullptr);

    //see ViewParamModel::initNode()
    QStandardItem* inputsItem = viewParams->getInputs();
    QStandardItem* paramsItem = viewParams->getParams();
    QStandardItem* outputsItem = viewParams->getOutputs();
    QStandardItem* legacyInputs = viewParams->getLegacyInputs();
    QStandardItem* legacyParams = viewParams->getLegacyParams();
    QStandardItem* legacyOutputs = viewParams->getLegacyOutputs();

    connect(viewParams, &QStandardItemModel::rowsInserted, this, &ZenoNode::onViewParamInserted);
    connect(viewParams, &QStandardItemModel::rowsAboutToBeRemoved, this, &ZenoNode::onViewParamAboutToBeRemoved);
    connect(viewParams, &QStandardItemModel::dataChanged, this, &ZenoNode::onViewParamDataChanged);
    connect(viewParams, &QStandardItemModel::rowsAboutToBeMoved, this, &ZenoNode::onViewParamAboutToBeMoved);
    connect(viewParams, &QStandardItemModel::rowsMoved, this, &ZenoNode::onViewParamsMoved);

    //params.
    m_paramsLayout = initParams(paramsItem, pScene);
    m_bodyLayout->addLayout(m_paramsLayout);

    if (m_paramsLayout->count() > 0)
        m_bodyLayout->addSpacing(24, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

    m_inputsLayout = initSockets(inputsItem, legacyInputs, true, pScene);
    m_bodyLayout->addLayout(m_inputsLayout);

    m_outputsLayout = initSockets(outputsItem, legacyOutputs, false, pScene);
    m_bodyLayout->addLayout(m_outputsLayout);

    m_dirtyMarker = new ZLayoutBackground;
    m_dirtyMarker->setColors(false, QColor(0, 0, 0, 0));
    m_dirtyMarker->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_dirtyMarker->setGeometry(QRectF(0, 0, 1, 3));

    m_bodyLayout->addSpacing(13, QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));
    m_bodyLayout->addItem(m_dirtyMarker);

    bodyWidget->setLayout(m_bodyLayout);
    return bodyWidget;
}

ZGraphicsLayout* ZenoNode::initCustomParamWidgets()
{
    return nullptr;
}

QGraphicsItem* ZenoNode::initParamWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx)
{
    const PARAM_CONTROL ctrl = (PARAM_CONTROL)paramIdx.data(ROLE_PARAM_CTRL).toInt();
    if (ctrl == CONTROL_NONVISIBLE)
        return nullptr;

    const QPersistentModelIndex perIdx(paramIdx);
    bool bFloat = ctrl == CONTROL_VEC2_FLOAT || ctrl == CONTROL_VEC3_FLOAT || ctrl == CONTROL_VEC4_FLOAT || ctrl == CONTROL_FLOAT;
    Callback_EditFinished cbUpdateParam = [=](QVariant newValue) {
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        if (!pModel)
            return;
        if (bFloat)
        {
            AppHelper::updateCurve(paramIdx.data(ROLE_PARAM_VALUE), newValue);
        }
        pModel->ModelSetData(perIdx, newValue, ROLE_PARAM_VALUE);
    };
    auto cbUpdateSocketDefldWithSlider = [=](QVariant newValue) {
        AppHelper::modifyOptixObjDirectly(newValue, m_index, perIdx);
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    auto cbGetIndexData = [=]() -> QVariant { 
        return paramIdx.data(ROLE_PARAM_VALUE);
    };

    auto cbGetZsgDir = []() {
        QString path = zenoApp->graphsManagment()->zsgPath();
        if (path.isEmpty())
            return QString("");
        QFileInfo fi(path);
        QString dirPath = fi.dir().path();
        return dirPath;
    };

    CallbackCollection cbSet;
    cbSet.cbEditFinished = cbUpdateParam;
    cbSet.cbEditFinishedWithSlider = cbUpdateSocketDefldWithSlider;
    cbSet.cbSwitch = cbSwith;
    cbSet.cbGetIndexData = cbGetIndexData;
    cbSet.cbGetZsgDir = cbGetZsgDir;

    const QString& paramName = paramIdx.data(ROLE_PARAM_NAME).toString();
    QVariant deflValue = paramIdx.data(ROLE_PARAM_VALUE);
    const QString& typeDesc = paramIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& ctrlProps = paramIdx.data(ROLE_VPARAM_CTRL_PROPERTIES);
    //key frame
    bool bKeyFrame = false;
    if (bFloat)
    {
        bKeyFrame = AppHelper::getCurveValue(deflValue);
    }
    QGraphicsItem* pControl = zenoui::createItemWidget(deflValue, ctrl, typeDesc, cbSet, scene, ctrlProps);

    if (bFloat)
    {
        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        ZASSERT_EXIT(mainWin, pControl);
        ZTimeline* timeline = mainWin->timeline();
        ZASSERT_EXIT(timeline, pControl);
        QObject* context = nullptr;
        if (QGraphicsProxyWidget* pWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pControl))
            context = pWidget;
        else if (QGraphicsTextItem* pTextItem = qgraphicsitem_cast<QGraphicsTextItem*>(pControl))
            context = pTextItem;
        if (!context)
            return pControl;
        onUpdateFrame(pControl, timeline->value(), paramIdx.data(ROLE_PARAM_VALUE));
        connect(timeline, &ZTimeline::sliderValueChanged, context, [=](int nFrame) {
            onUpdateFrame(pControl, nFrame, paramIdx.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
        connect(mainWin, &ZenoMainWindow::visFrameUpdated, context, [=](bool bGLView, int nFrame) {
            onUpdateFrame(pControl, nFrame, paramIdx.data(ROLE_PARAM_VALUE));
            }, Qt::UniqueConnection);
    }
    return pControl;
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
        QString custName = m_index.data(ROLE_CUSTOM_OBJNAME).toString();
        if (custName.isEmpty())
        {
            m_NameItem->setText(newName);
            ZGraphicsLayout::updateHierarchy(m_NameItem);
        }
        else
        {
            QString text = m_pCategoryItem->text();
            m_pCategoryItem->setText(text.left(text.indexOf(":") + 1) + newName);
            ZGraphicsLayout::updateHierarchy(m_pCategoryItem);
        }
    }
}

ZSocketLayout* ZenoNode::getSocketLayout(bool bInput, const QString& name)
{
    if (bInput)
    {
        for (int i = 0; i < m_inSockets.size(); i++)
        {
            QModelIndex idx = m_inSockets[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
                return m_inSockets[i];
        }
    }
    else
    {
        for (int i = 0; i < m_outSockets.size(); i++)
        {
            QModelIndex idx = m_outSockets[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
                return m_outSockets[i];
        }
    }
    return nullptr;
}

bool ZenoNode::removeSocketLayout(bool bInput, const QString& name)
{
    if (bInput)
    {
        for (int i = 0; i < m_inSockets.size(); i++)
        {
            QModelIndex idx = m_inSockets[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
            {
                m_inSockets.remove(i);
                return false;
            }
        }
    }
    else
    {
        for (int i = 0; i < m_outSockets.size(); i++)
        {
            QModelIndex idx = m_outSockets[i]->viewSocketIdx();
            QString sockName = idx.data(ROLE_PARAM_NAME).toString();
            if (sockName == name)
            {
                m_outSockets.remove(i);
                return false;
            }
        }
    }
    return false;
}

void ZenoNode::onViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (roles.isEmpty())
        return;

    if (!m_index.isValid())
        return;

    QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
    if (!viewParams)
        return;

    QStandardItem* pItem = viewParams->itemFromIndex(topLeft);
    ZASSERT_EXIT(pItem);
    int vType = pItem->data(ROLE_VPARAM_TYPE).toInt();
    if (vType != VPARAM_PARAM)
    {
        return;
    }

    int role = roles[0];
    if (role != ROLE_PARAM_NAME 
        && role != ROLE_PARAM_VALUE 
        && role != ROLE_PARAM_CTRL 
        && role != ROLE_VPARAM_CTRL_PROPERTIES
        && role != ROLE_VPARAM_TOOLTIP
        && role != ROLE_PARAM_NETLABEL)
        return;

    QModelIndex viewParamIdx = pItem->index();

    QStandardItem* parentItem = pItem->parent();
    ZASSERT_EXIT(parentItem);
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    const QString& groupName = parentItem->text();
    const QString& paramName = pItem->data(ROLE_PARAM_NAME).toString();

    if (role == ROLE_PARAM_NAME || role == ROLE_VPARAM_TOOLTIP)
    {
        const int paramCtrl = pItem->data(ROLE_PARAM_CTRL).toInt();
        if (groupName == iotags::params::node_inputs)
        {
            for (int i = 0; i < m_inSockets.size(); i++)
            {
                ZSocketLayout* pSocketLayout = m_inSockets[i];
                QModelIndex socketIdx = pSocketLayout->viewSocketIdx();
                if (socketIdx == viewParamIdx)
                {
                    if (role == ROLE_PARAM_NAME)
                        pSocketLayout->updateSockName(paramName);   //only update name on control.
                    else if (role == ROLE_VPARAM_TOOLTIP)
                        pSocketLayout->updateSockNameToolTip(pItem->data(ROLE_VPARAM_TOOLTIP).toString()); 
                    break;
                }
            }
        }
        else if (groupName == iotags::params::node_params)
        {
            for (auto it = m_params.begin(); it != m_params.end(); it++)
            {
                if (it->second.viewidx == viewParamIdx)
                {
                    if (role == ROLE_PARAM_NAME) 
                    {
                        QString oldName = it->first;
                        it->first = paramName;
                        it->second.param_name->setText(paramName);
                    }
                    else if (role == ROLE_VPARAM_TOOLTIP) 
                    {
                        it->second.param_name->setToolTip(pItem->data(ROLE_VPARAM_TOOLTIP).toString());
                    }
                    break;
                }
            }
        }
        else if (groupName == iotags::params::node_outputs)
        {
            for (int i = 0; i < m_outSockets.size(); i++)
            {
                ZSocketLayout* pSocketLayout = m_outSockets[i];
                QModelIndex socketIdx = pSocketLayout->viewSocketIdx();
                if (socketIdx == viewParamIdx)
                {
                    if (role == ROLE_PARAM_NAME)
                        pSocketLayout->updateSockName(paramName);
                    else if (role == ROLE_VPARAM_TOOLTIP)
                        pSocketLayout->updateSockNameToolTip(pItem->data(ROLE_VPARAM_TOOLTIP).toString()); 
                }
            }
        }
        updateWhole();
        return;
    }

    if (groupName == iotags::params::node_inputs)
    {
        ZSocketLayout* pControlLayout = getSocketLayout(true, paramName);
        QGraphicsItem* pControl = nullptr;
        if (pControlLayout)
            pControl = pControlLayout->control();

        PARAM_CONTROL ctrl = (PARAM_CONTROL)pItem->data(ROLE_PARAM_CTRL).toInt();
        switch (role)
        {
            case ROLE_PARAM_CTRL:
            {
                PARAM_CONTROL oldCtrl = pControl ? (PARAM_CONTROL)pControl->data(GVKEY_CONTROL).toInt() : CONTROL_NONE;
                if (ctrl != oldCtrl)
                {
                    QGraphicsItem* pNewControl = initSocketWidget(pScene, pItem->index());
                    pControlLayout->setControl(pNewControl);
                    if (pNewControl)
                        pNewControl->setVisible(pControlLayout->socketItem()->sockStatus() != ZenoSocketItem::STATUS_CONNECTED);
                    pControl = pNewControl;
                    updateWhole();
                }
                //set value on pControl.
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, ctrl, deflValue, pScene);
                break;
            }
            case ROLE_PARAM_VALUE:
            {
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, ctrl, deflValue, pScene);
                if (CONTROL_VEC4_FLOAT == ctrl || CONTROL_VEC3_FLOAT == ctrl 
                    || CONTROL_VEC2_FLOAT == ctrl || CONTROL_FLOAT == ctrl)
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
            case ROLE_VPARAM_CTRL_PROPERTIES: 
            {
                QVariant value = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES);
                ZenoGvHelper::setCtrlProperties(pControl, value);
                const QVariant &deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, ctrl, deflValue, pScene);
                break;
            }
            case ROLE_PARAM_NETLABEL:
            {
                if (pControlLayout && pControlLayout->socketItem()) {
                    ZenoSocketItem* pSocketItem = pControlLayout->socketItem();
                    const QString& lblName = pItem->data(ROLE_PARAM_NETLABEL).toString();
                    pSocketItem->onNetLabelChanged(lblName);

                    //hide or visible control.
                    QString sockName = viewParamIdx.data(ROLE_PARAM_NAME).toString();
                    ZSocketLayout* pSocketLayout = getSocketLayout(true, sockName);
                    if (pSocketLayout && pSocketLayout->control())
                    {
                        pSocketLayout->control()->setVisible(lblName.isEmpty());
                        updateWhole();
                    }
                }
                break;
            }
        }
    }
    else if (groupName == iotags::params::node_params)
    {
        const QString& sockName = viewParamIdx.data(ROLE_VPARAM_NAME).toString();
        const QString& newType = viewParamIdx.data(ROLE_PARAM_TYPE).toString();
        PARAM_CONTROL ctrl = (PARAM_CONTROL)viewParamIdx.data(ROLE_PARAM_CTRL).toInt();
        ZASSERT_EXIT(m_params.find(sockName) != m_params.end());
        const auto& paramCtrl = m_params[sockName];

        switch (role)
        {
            case ROLE_PARAM_CTRL:
            {
                ZGraphicsLayout* pParamLayout = paramCtrl.ctrl_layout;
                pParamLayout->removeItem(paramCtrl.param_control);

                QGraphicsItem* pNewControl = initParamWidget(pScene, viewParamIdx);
                if (pNewControl)
                {
                    pParamLayout->addItem(pNewControl);
                    m_params[sockName].param_control = pNewControl;
                }
                else
                {
                    m_params[sockName].param_control = nullptr;
                }
                updateWhole();
                break;
            }
            case ROLE_PARAM_VALUE:
            {
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(paramCtrl.param_control, ctrl, deflValue, pScene);
                break;
            }
            case ROLE_VPARAM_CTRL_PROPERTIES: 
            {
                QVariant value = pItem->data(ROLE_VPARAM_CTRL_PROPERTIES);
                ZenoGvHelper::setCtrlProperties(paramCtrl.param_control, value);
                break;
            }
        }
    }
    else if (groupName == iotags::params::node_outputs)
    {
        ZSocketLayout* pControlLayout = getSocketLayout(false, paramName);
        if (ROLE_PARAM_NETLABEL == role && pControlLayout && pControlLayout->socketItem())
        {
            ZenoSocketItem* pSocketItem = pControlLayout->socketItem();
            const QString& lblName = pItem->data(ROLE_PARAM_NETLABEL).toString();
            pSocketItem->onNetLabelChanged(lblName);
        }
    }
}

void ZenoNode::onViewParamInserted(const QModelIndex& parent, int first, int last)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;
    QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);

    if (!parent.isValid())
    {
        QStandardItem* _root = viewParams->invisibleRootItem();
        ZASSERT_EXIT(_root && _root->rowCount() == 1);

        QStandardItem* pRoot = _root->child(0);
        ZASSERT_EXIT(pRoot && pRoot->rowCount() == 1);

        QStandardItem* pTab = pRoot->child(0);
        for (int r = 0; r < pTab->rowCount(); r++)
        {
            QStandardItem* pGroup = pTab->child(r);
            if (!pGroup) return;

            const QString& groupName = pGroup->text();
            if (groupName == iotags::params::node_params)
            {
                ZGraphicsLayout* playout = initParams(pGroup, pScene);
                if (m_paramsLayout)
                {
                    m_paramsLayout->clear();
                    m_bodyLayout->removeLayout(m_paramsLayout);
                }
                m_paramsLayout = playout;
                m_bodyLayout->insertLayout(0, m_paramsLayout);
                updateWhole();
            }
            else if (groupName == iotags::params::node_inputs)
            {
                ZGraphicsLayout* playout = initSockets(pGroup, nullptr, true, pScene);
                if (m_inputsLayout)
                {
                    m_inputsLayout->clear();
                    m_bodyLayout->removeLayout(m_inputsLayout);
                }
                m_inputsLayout = playout;
                m_bodyLayout->insertLayout(1, m_inputsLayout);
                updateWhole();
            }
            else if (groupName == iotags::params::node_outputs)
            {
                ZGraphicsLayout* playout = initSockets(pGroup, nullptr, false, pScene);
                if (m_outputsLayout)
                {
                    m_outputsLayout->clear();
                    m_bodyLayout->removeLayout(m_outputsLayout);
                }
                m_outputsLayout = playout;
                m_bodyLayout->insertLayout(2, m_outputsLayout);
                updateWhole();
            }
        }
        return;
    }

    QStandardItem* parentItem = viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem && parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    QStandardItem* paramItem = parentItem->child(first);
    const QString& groupName = parentItem->text();
    const QModelIndex& viewParamIdx = paramItem->index();

    //see ViewParamModel::initNode
    if (groupName == iotags::params::node_inputs || groupName == iotags::params::node_outputs)
    {
        bool bInput = groupName == iotags::params::node_inputs;
        ZGraphicsLayout* pSocketsLayout = bInput ? m_inputsLayout : m_outputsLayout;        
        ZSocketLayout *pSocketLayout = addSocket(viewParamIdx, bInput, pScene);
        ZASSERT_EXIT(pSocketLayout);
        pSocketsLayout->addLayout(pSocketLayout);
        updateWhole();
    }
    else if (groupName == iotags::params::node_params)
    {
        ZASSERT_EXIT(m_paramsLayout);
        m_paramsLayout->addLayout(addParam(viewParamIdx, pScene));
    }
}

void ZenoNode::onViewParamAboutToBeMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int row)
{

}

void ZenoNode::onViewParamsMoved(const QModelIndex& parent, int start, int end, const QModelIndex& destination, int destRow)
{
    QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);
    QStandardItem* parentItem = viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem && parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    if (parent != destination || start == destRow)
        return;

    const QString& groupName = parentItem->text();
    if (groupName == iotags::params::node_inputs)
    {
#if 0
        //test socket index in each elem from m_inSockets.
        for (int i = 0; i < m_inSockets.size(); i++)
        {
            ZSocketLayout* pSocketLayout = m_inSockets[i];
            QModelIndex socketIdx = pSocketLayout->viewSocketIdx();
            if (socketIdx.isValid()) {
                const QString sockName = socketIdx.data(ROLE_PARAM_NAME).toString();
                int j;
                j = 0;
            }
        }
#endif
        //m_inSockets.move(start, destRow);
        ZASSERT_EXIT(m_inputsLayout);
        m_inputsLayout->moveItem(start, destRow);
        updateWhole();
    }
    else if (groupName == iotags::params::node_outputs)
    {
        //m_outSockets.move(start, destRow);
        ZASSERT_EXIT(m_outputsLayout);
        m_outputsLayout->moveItem(start, destRow);
        updateWhole();
    }
}

void ZenoNode::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (!parent.isValid())
    {
        //remove all component.
        m_paramsLayout->clear();
        m_inputsLayout->clear();
        m_outputsLayout->clear();
        m_params.clear();
        m_inSockets.clear();
        m_outSockets.clear();
        return;
    }

    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!m_index.isValid())
        return;

    QStandardItemModel* viewParams = QVariantPtr<QStandardItemModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
    ZASSERT_EXIT(viewParams);

    QStandardItem* parentItem = viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem && parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    QStandardItem* paramItem = parentItem->child(first);
    const QString& groupName = parentItem->text();
    const QModelIndex& viewParamIdx = paramItem->index();
    const int paramCtrl = viewParamIdx.data(ROLE_PARAM_CTRL).toInt();
    if (groupName == iotags::params::node_inputs || groupName == iotags::params::node_outputs)
    {
        bool bInput = groupName == iotags::params::node_inputs;
        const QString& sockName = viewParamIdx.data(ROLE_VPARAM_NAME).toString();
        ZSocketLayout* pSocketLayout = getSocketLayout(bInput, sockName);
        removeSocketLayout(bInput, sockName);

        ZASSERT_EXIT(pSocketLayout);
        ZGraphicsLayout* pParentLayout = pSocketLayout->parentLayout();
        pParentLayout->removeLayout(pSocketLayout);
        updateWhole();
    }
    else if (groupName == iotags::params::node_params)
    {
        const QString& paramName = viewParamIdx.data(ROLE_PARAM_NAME).toString();

        ZASSERT_EXIT(m_params.find(paramName) != m_params.end());
        ZGraphicsLayout* paramLayout = m_params[paramName].ctrl_layout;
        m_params.remove(paramName);
        ZASSERT_EXIT(paramLayout);
        ZGraphicsLayout* pParentLayout = paramLayout->parentLayout();
        if (pParentLayout)
            pParentLayout->removeLayout(paramLayout);
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
        pView->focusOn(nodeIdx.data(ROLE_OBJID).toString(), QPointF(), false);
    }
}

ZGraphicsLayout* ZenoNode::initSockets(QStandardItem* socketItems, QStandardItem* legacyItems, const bool bInput, ZenoSubGraphScene* pScene)
{
    ZASSERT_EXIT(socketItems, nullptr);

    ZGraphicsLayout* pSocketsLayout = new ZGraphicsLayout(false);
    pSocketsLayout->setSpacing(5);

    for (int r = 0; r < socketItems->rowCount(); r++)
    {
        const QStandardItem* pItem = socketItems->child(r);
        const QModelIndex& viewIdx = pItem->index();
        ZSocketLayout *pSocketLayout = addSocket(viewIdx, bInput, pScene);
        pSocketsLayout->addLayout(pSocketLayout);
    }
    if (legacyItems)
    {
        for (int r = 0; r < legacyItems->rowCount(); r++)
        {
            const QStandardItem* pItem = legacyItems->child(r);
            const QModelIndex& viewIdx = pItem->index();
            ZSocketLayout* pSocketLayout = addSocket(viewIdx, bInput, pScene);
            pSocketsLayout->addLayout(pSocketLayout);
        }
    }
    return pSocketsLayout;
}

ZSocketLayout* ZenoNode::addSocket(const QModelIndex& viewSockIdx, bool bInput, ZenoSubGraphScene* pScene)
{
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();

    QPersistentModelIndex perSockIdx = viewSockIdx;

    CallbackForSocket cbSocket;
    cbSocket.cbOnSockClicked = [=](ZenoSocketItem* pSocketItem, Qt::MouseButton button) {
        emit socketClicked(pSocketItem, button);
    };
    cbSocket.cbOnSockLayoutChanged = [=]() {
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    };
    cbSocket.cbOnSockNetlabelClicked = [=](QString netlabel) {
        if (perSockIdx.isValid()) {
            bool bInput = PARAM_INPUT == perSockIdx.data(ROLE_PARAM_CLASS);
            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
            QMenu* menu = nullptr;
            if (bInput)
            {
                QModelIndex outSock = pModel->getNetOutput(m_subGpIndex, netlabel);
                ZASSERT_EXIT(outSock.isValid());
                QString objPath = outSock.data(ROLE_OBJPATH).toString();
                //jump directly.
                QMenu* menu = new QMenu;
                QString tip = tr("jump to %1").arg(objPath);
                QAction* pAction = new QAction(tip);
                connect(pAction, &QAction::triggered, this, [=]() {
                    QModelIndex nodeIdx = outSock.data(ROLE_NODE_IDX).toModelIndex();
                    focusOnNode(nodeIdx);
                });
                menu->addAction(pAction);
                menu->exec(QCursor::pos());
                menu->deleteLater();
            }
            else
            {
                //find all input socks.
                QList<QModelIndex> inSocks = pModel->getNetInputs(m_subGpIndex, netlabel);
                if (inSocks.isEmpty())
                    return;

                QMenu* menu = new QMenu;
                for (auto insock : inSocks)
                {
                    QString objPath = insock.data(ROLE_OBJPATH).toString();
                    QAction* pAction = new QAction(objPath);
                    connect(pAction, &QAction::triggered, this, [=]() {
                        QModelIndex nodeIdx = insock.data(ROLE_NODE_IDX).toModelIndex();
                        focusOnNode(nodeIdx);
                    });
                    menu->addAction(pAction);
                }
                menu->exec(QCursor::pos());
                menu->deleteLater();
            }
        }
    };
    cbSocket.cbOnSockNetlabelEdited = [=](ZenoSocketItem* pSocketItem) {
        const QString& label = pSocketItem->netLabel();
        QModelIndex sockIdx = pSocketItem->paramIndex();
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        const QString& oldName = sockIdx.data(ROLE_PARAM_NETLABEL).toString();
        //check label
        if (oldName == label)
            return;
        QStringList labels = pModel->dumpLabels(m_subGpIndex);
        if (labels.contains(label))
        {
            QMessageBox::warning(nullptr, tr("Warring"), tr("Net label invalid!"));
            pSocketItem->onNetLabelChanged(oldName);
            return;
        }
        pModel->updateNetLabel(m_subGpIndex, sockIdx, oldName, label, true);

        auto procClipbrd = zenoApp->procClipboard();
        ZASSERT_EXIT(procClipbrd);
        procClipbrd->setCopiedAddress("");
    };
    cbSocket.cbActionTriggered = [=](QAction* pAction, const QModelIndex& socketIdx) {
        QString text = pAction->text();
        if (pAction->text() == tr("Delete Net Label")) {
            IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
            ZASSERT_EXIT(pModel);
            pModel->removeNetLabel(m_subGpIndex, socketIdx);
        }
    };

    const QString& sockName = viewSockIdx.data(ROLE_VPARAM_NAME).toString();
    PARAM_CONTROL ctrl = (PARAM_CONTROL)viewSockIdx.data(ROLE_PARAM_CTRL).toInt();
    const QString& sockType = viewSockIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& deflVal = viewSockIdx.data(ROLE_PARAM_VALUE);
    const PARAM_LINKS& links = viewSockIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
    int sockProp = viewSockIdx.data(ROLE_PARAM_SOCKPROP).toInt();

    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());
    bool bSocketEnable = true;
    if (SOCKPROP_LEGACY == viewSockIdx.data(ROLE_PARAM_SOCKPROP) || type == NO_VERSION_NODE)
    {
        bSocketEnable = false;
    }

    ZSocketLayout* pMiniLayout = nullptr;
    if (sockProp & SOCKPROP_DICTLIST_PANEL) {
        pMiniLayout = new ZDictSocketLayout(pModel, viewSockIdx, bInput);
    } 
    else if (sockProp & SOCKPROP_GROUP_LINE) {
        pMiniLayout = new ZGroupSocketLayout(pModel, viewSockIdx, bInput);
    }
    else {
        pMiniLayout = new ZSocketLayout(pModel, viewSockIdx, bInput);
        qreal margin = ZenoStyle::dpiScaled(16);
        if (bInput)
            pMiniLayout->setContentsMargin(0, 0, 0, margin);
    }
    pMiniLayout->initUI(pModel, cbSocket);
    pMiniLayout->setDebugName(sockName);

    if (bInput)
    {
        QGraphicsItem* pSocketControl = initSocketWidget(pScene, viewSockIdx);
        pMiniLayout->setControl(pSocketControl);
        if (pSocketControl) {
            const QString& netLabel = viewSockIdx.data(ROLE_PARAM_NETLABEL).toString();
            pSocketControl->setVisible((links.isEmpty() && netLabel.isEmpty()) ||
                (nodeName() == "GenerateCommands" && sockName == "source"));
            pSocketControl->setEnabled(bSocketEnable);
        }
    }

    if (bInput)
        m_inSockets.append(pMiniLayout);
    else
        m_outSockets.append(pMiniLayout);

    return pMiniLayout;
}

ZGraphicsLayout* ZenoNode::initParams(QStandardItem* paramItems, ZenoSubGraphScene* pScene)
{
    ZASSERT_EXIT(paramItems, nullptr);

    ZGraphicsLayout* paramsLayout = new ZGraphicsLayout(false);
    paramsLayout->setSpacing(5);
    qreal margin = ZenoStyle::dpiScaled(16);
    paramsLayout->setContentsMargin(0, margin, 0, margin);

    for (int r = 0; r < paramItems->rowCount(); r++)
    {
        const QStandardItem* pItem = paramItems->child(r);
        const QModelIndex& idx = pItem->index();
        paramsLayout->addLayout(addParam(idx, pScene));
    }

    ZGraphicsLayout* pCustomParams = initCustomParamWidgets();
    if (pCustomParams)
    {
        paramsLayout->addLayout(pCustomParams);
    }
    return paramsLayout;
}

ZGraphicsLayout* ZenoNode::addParam(const QModelIndex& viewparamIdx, ZenoSubGraphScene* pScene)
{
    const QString& paramName = viewparamIdx.data(ROLE_VPARAM_NAME).toString();
    PARAM_CONTROL ctrl = (PARAM_CONTROL)viewparamIdx.data(ROLE_PARAM_CTRL).toInt();
    const QString& paramType = viewparamIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& value = viewparamIdx.data(ROLE_PARAM_VALUE);
    
    if (ctrl == CONTROL_NONVISIBLE)
        return nullptr;

    _param_ctrl paramCtrl;
    paramCtrl.ctrl_layout = new ZGraphicsLayout(true);
    paramCtrl.ctrl_layout->setSpacing(ZenoStyle::dpiScaled(32));
    auto textItem = new ZSimpleTextItem(paramName);
    textItem->setBrush(m_renderParams.socketClr.color());
    textItem->setFont(m_renderParams.socketFont);
    textItem->updateBoundingRect();
    textItem->setToolTip(viewparamIdx.data(ROLE_VPARAM_TOOLTIP).toString());
    paramCtrl.param_name = textItem;
    paramCtrl.viewidx = viewparamIdx;
    paramCtrl.ctrl_layout->addItem(paramCtrl.param_name, Qt::AlignVCenter);

    switch (ctrl)
    {
        case CONTROL_STRING:
        case CONTROL_INT:
        case CONTROL_FLOAT:
        case CONTROL_BOOL:
        case CONTROL_VEC2_FLOAT:
        case CONTROL_VEC2_INT:
        case CONTROL_VEC3_FLOAT:
        case CONTROL_VEC3_INT:
        case CONTROL_VEC4_FLOAT:
        case CONTROL_VEC4_INT:
        case CONTROL_ENUM:
        case CONTROL_READPATH:
        case CONTROL_WRITEPATH:
        case CONTROL_MULTILINE_STRING:
        case CONTROL_COLOR_VEC3F:
        case CONTROL_CURVE:
        case CONTROL_HSLIDER:
        case CONTROL_HSPINBOX:
        case CONTROL_HDOUBLESPINBOX:
        case CONTROL_SPINBOX_SLIDER:
        case CONTROL_DIRECTORY:
        {
            QGraphicsItem* pWidget = initParamWidget(pScene, viewparamIdx);
            paramCtrl.ctrl_layout->addItem(pWidget, Qt::AlignRight | Qt::AlignVCenter);
            paramCtrl.param_control = pWidget;
            break;
        }
        default:
        {
            break;
        }
    }

    m_params[paramName] = paramCtrl;
    return paramCtrl.ctrl_layout;
}

Callback_OnButtonClicked ZenoNode::registerButtonCallback(const QModelIndex& paramIdx)
{
    return []() {

    };
}

QGraphicsItem* ZenoNode::initSocketWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx)
{
    const QPersistentModelIndex perIdx(paramIdx);

    PARAM_CONTROL ctrl = (PARAM_CONTROL)paramIdx.data(ROLE_PARAM_CTRL).toInt();
    bool bFloat = ctrl == CONTROL_VEC2_FLOAT || ctrl == CONTROL_VEC3_FLOAT || ctrl == CONTROL_VEC4_FLOAT || ctrl == CONTROL_FLOAT;
    auto cbUpdateSocketDefl = [=](QVariant newValue) {
        if (bFloat)
        {
            if (!AppHelper::updateCurve(paramIdx.data(ROLE_PARAM_VALUE), newValue))
            {
                onViewParamDataChanged(perIdx, perIdx, QVector<int>() << ROLE_PARAM_VALUE);
                return;
            }
        }
        AppHelper::socketEditFinished(newValue, m_index, perIdx);
    };
    auto cbUpdateSocketDefldWithSlider = [=](QVariant newValue) {
        AppHelper::modifyOptixObjDirectly(newValue, m_index, perIdx);
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    const QString& sockType = paramIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& deflVal = paramIdx.data(ROLE_PARAM_VALUE);
    const QVariant& ctrlProps = paramIdx.data(ROLE_VPARAM_CTRL_PROPERTIES);

    auto cbGetIndexData = [=]() -> QVariant { 
        return perIdx.data(ROLE_PARAM_VALUE);
    };

    auto cbGetZsgDir = []() {
        QString path = zenoApp->graphsManagment()->zsgPath();
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

void ZenoNode::onSocketLinkChanged(const QModelIndex& paramIdx, bool bInput, bool bAdded)
{
    ZenoSocketItem* pSocket = getSocketItem(paramIdx);
    if (pSocket == nullptr)
        return;

    QModelIndex idx = pSocket->paramIndex();
    // the removal of links from socket is executed before the removal of link itself.
    PARAM_LINKS links = idx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
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
        if (this->nodeName() == "GenerateCommands" && sockName == "source")
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
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    QColor clrHeaderBg;
    if (m_bError)
        clrHeaderBg = QColor(200, 84, 79);
    else if (pGraphsModel->IsSubGraphNode(m_index))
        clrHeaderBg = QColor("#1D5F51");
    else
        clrHeaderBg = m_renderParams.headerBg.clr_normal;
    m_headerWidget->setColors(false, clrHeaderBg, QColor(), QColor());
    update();
}

ZenoSocketItem* ZenoNode::getSocketItem(const QModelIndex& sockIdx)
{
    for (ZSocketLayout* socklayout : m_inSockets)
    {
        if (ZenoSocketItem* pItem = socklayout->socketItemByIdx(sockIdx))
        {
            return pItem;
        }
    }
    for (ZSocketLayout* socklayout : m_outSockets)
    {
        if (ZenoSocketItem* pItem = socklayout->socketItemByIdx(sockIdx))
            return pItem;
    }
    return nullptr;
}

ZenoSocketItem* ZenoNode::getNearestSocket(const QPointF& pos, bool bInput)
{
    ZenoSocketItem* pItem = nullptr;
    float minDist = std::numeric_limits<float>::max();
    auto socks = bInput ? m_inSockets : m_outSockets;
    for (ZSocketLayout* sock : socks)
    {
        //todo: socket now is a children of sockettext.
        ZenoSocketItem* pSocketItem = sock->socketItem();
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
    for (ZSocketLayout* sock : m_inSockets)
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
    for (ZSocketLayout* sock : m_outSockets)
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

QPointF ZenoNode::getSocketPos(const QModelIndex& sockIdx)
{
    ZASSERT_EXIT(sockIdx.isValid(), QPointF());

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped)
    {
        PARAM_CLASS coreCls = (PARAM_CLASS)sockIdx.data(ROLE_PARAM_CLASS).toInt();
        QRectF rc = m_headerWidget->sceneBoundingRect();
        if (coreCls == PARAM_INPUT || coreCls == PARAM_INNER_INPUT || coreCls == PARAM_LEGACY_INPUT) {
            return QPointF(rc.left(), rc.center().y());
        } else if (coreCls == PARAM_OUTPUT || coreCls == PARAM_INNER_OUTPUT || coreCls == PARAM_LEGACY_OUTPUT) {
            return QPointF(rc.right(), rc.center().y());
        } else {
            return QPointF(0, 0);
        }
    }
    else
    {
        PARAM_CLASS cls = (PARAM_CLASS)sockIdx.data(ROLE_PARAM_CLASS).toInt();
        if (cls == PARAM_INNER_INPUT || cls == PARAM_INPUT || cls == PARAM_LEGACY_INPUT)
        {
            for (ZSocketLayout* socklayout : m_inSockets)
            {
                bool exist = false;
                QPointF pos = socklayout->getSocketPos(sockIdx, exist);
                if (exist)
                    return pos;
            }
        }
        else if (cls == PARAM_INNER_OUTPUT || cls == PARAM_OUTPUT || cls == PARAM_LEGACY_OUTPUT)
        {
            for (ZSocketLayout* socklayout : m_outSockets)
            {
                bool exist = false;
                QPointF pos = socklayout->getSocketPos(sockIdx, exist);
                if (exist)
                    return pos;
            }
        }
        zeno::log_warn("socket pos error");
        return QPointF(0, 0);
    }
}

QString ZenoNode::nodeId() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_OBJID).toString();
}

QString ZenoNode::nodeName() const
{
    ZASSERT_EXIT(m_index.isValid(), "");
    return m_index.data(ROLE_OBJNAME).toString();
}

QPointF ZenoNode::nodePos() const
{
    ZASSERT_EXIT(m_index.isValid(), QPointF());
    return m_index.data(ROLE_OBJPOS).toPointF();
}

void ZenoNode::updateNodePos(const QPointF &pos, bool enableTransaction) 
{
    QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();
    if (oldPos == pos)
        return;
    STATUS_UPDATE_INFO info;
    info.role = ROLE_OBJPOS;
    info.newValue = pos;
    info.oldValue = oldPos;
    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    pGraphsModel->updateBlackboard(nodeId(), QVariant::fromValue(info), m_subGpIndex, enableTransaction);
    m_bMoving = false;
}

void ZenoNode::onUpdateParamsNotDesc()
{
}

void ZenoNode::onMarkDataChanged(bool bDirty)
{
    QColor clrMarker;
    if (bDirty && bShowDataChanged)
    {
        clrMarker = QColor(240, 215, 4);
    }
    else
    {
        clrMarker = QColor(0, 0, 0, 0);
    }
    ZASSERT_EXIT(m_dirtyMarker);
    m_dirtyMarker->setColors(false, clrMarker);
    updateWhole();
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
        for (auto item : m_inSockets) {
            item->setVisible(bVisible);
        }
        for (auto item : m_outSockets) {
            item->setVisible(bVisible);
        }
        for (auto it = m_params.begin(); it != m_params.end(); it++) {
            if (it->second.param_control)
                it->second.param_control->setVisible(bVisible);
            if (it->second.param_name)
                it->second.param_name->setVisible(bVisible);
        }
        if (m_NameItem) {
            m_NameItem->setVisible(bVisible);
        }
        if (m_pCategoryItem) {
            m_pCategoryItem->setVisible(bVisible);
        }
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
        m_NameItemTip = new ZSimpleTextItem(m_NameItem->toPlainText(), this);
        m_NameItemTip->setBrush(QColor("#FFFFFF"));
        m_NameItemTip->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        m_NameItemTip->show();
    }
    if (m_NameItemTip) 
    {
        if (m_NameItemTip->text() != m_NameItem->toPlainText())
            m_NameItemTip->setText(m_NameItem->toPlainText());
        m_NameItemTip->setPos(QPointF(m_NameItem->pos().x(), -ZenoStyle::scaleWidth(36)));
    }
    if (m_bodyWidget)
        m_bodyWidget->setBorder(ZenoStyle::scaleWidth(2), QColor(18, 20, 22));
    m_dirtyMarker->resize(m_dirtyMarker->rect().width(), ZenoStyle::scaleWidth(3));
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
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    QPointF pos = event->pos();
    if (pGraphsModel && pGraphsModel->IsSubGraphNode(m_index))
    {
        scene()->clearSelection();
        this->setSelected(true);

        QMenu *nodeMenu = new QMenu;
        QAction *pCopy = new QAction("Copy");
        QAction *pDelete = new QAction("Delete");

        connect(pDelete, &QAction::triggered, this, [=]() {
            pGraphsModel->removeNode(m_index.data(ROLE_OBJID).toString(), m_subGpIndex, true);
        });

        nodeMenu->addAction(pCopy);
        nodeMenu->addAction(pDelete);
        QAction* pFork = new QAction("Fork");
        nodeMenu->addAction(pFork);
        connect(pFork, &QAction::triggered, this, [=]()
        {
            QModelIndex forkNode = pGraphsModel->fork(m_subGpIndex, index());
            if (forkNode.isValid())
            {
                ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
                if (pScene) {
                    pScene->select(forkNode.data(ROLE_OBJID).toString());
                }
            }
        });
        QAction* propDlg = new QAction(tr("Custom Param"));
        nodeMenu->addAction(propDlg);
        connect(propDlg, &QAction::triggered, this, [=]() {
            QStandardItemModel* params = QVariantPtr<QStandardItemModel>::asPtr(m_index.data(ROLE_NODE_PARAMS));
            ZASSERT_EXIT(params);
            ZEditParamLayoutDlg dlg(params, true, m_index, pGraphsModel);
            dlg.exec();
        });

        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else if (m_index.data(ROLE_OBJNAME).toString() == "BindMaterial")
    {
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
    }
    else
    {
        NODE_CATES cates = pGraphsModel->getCates();
        ZenoNewnodeMenu *menu = new ZenoNewnodeMenu(m_subGpIndex, cates, event->scenePos());
        menu->setEditorFocus();
        menu->exec(event->screenPos());
        menu->deleteLater();
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
                QString name = m_index.data(ROLE_OBJNAME).toString();
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
                QString name = m_index.data(ROLE_OBJNAME).toString();
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
    _base::mouseDoubleClickEvent(event);
    QList<QGraphicsItem*> wtf = scene()->items(event->scenePos());
    if (wtf.contains(m_NameItem))
    {
        QString name = m_index.data(ROLE_OBJNAME).toString();
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
    else if (wtf.contains(m_headerWidget))
    {
        onCollaspeBtnClicked();
    }
    else if (wtf.contains(m_bodyWidget))
    {
        const QString& name = nodeName();
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        QModelIndex subgIdx = pModel->index(name);
        if (subgIdx.isValid())
        {
            ZenoGraphsEditor* pEditor = getEditorViewByViewport(event->widget());
            if (pEditor)
            {
                pEditor->onPageActivated(subGraphIndex(), index());
            }
        }
        // for temp support to show handler via transform node
        else if (name.contains("TransformPrimitive"))
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
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        QPointF newPos = this->scenePos();
        QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();
        if (newPos != oldPos)
        {
            STATUS_UPDATE_INFO info;
            info.role = ROLE_OBJPOS;
            info.newValue = m_lastMoving;
            info.oldValue = oldPos;
            pGraphsModel->updateNodeStatus(nodeId(), info, m_subGpIndex, false);

            bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
            QRectF rc = m_headerWidget->sceneBoundingRect();

            emit inSocketPosChanged();
            emit outSocketPosChanged();
            //emit nodePosChangedSignal();

            m_lastMoving = QPointF();

            //other selected items also need update model data
            for (QGraphicsItem *item : this->scene()->selectedItems()) {
                if (item == this || !dynamic_cast<ZenoNode*>(item))
                    continue;
                ZenoNode *pNode = dynamic_cast<ZenoNode *>(item);
                info.newValue = pNode->scenePos();
                info.oldValue = pNode->index().data(ROLE_OBJPOS);
                pGraphsModel->updateNodeStatus(pNode->nodeId(), info, m_subGpIndex, false);
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
        const QString& ident = m_index.data(ROLE_OBJID).toString();
        pScene->collectNodeSelChanged(ident, bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        m_bMoving = true;
        ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(scene());
        bool isSnapGrid = ZenoSettingsManager::GetInstance().getValue(zsSnapGrid).toBool();
        if (pScene && isSnapGrid)
        {
            QPointF pos = value.toPointF();
            int x = pos.x(), y = pos.y();
            x = x - x % SCENE_GRID_SIZE;
            y = y - y % SCENE_GRID_SIZE;
            return QPointF(x, y);
        }
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        m_bMoving = true;
        m_lastMoving = value.toPointF();
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    }
    else if (change == ItemScenePositionHasChanged)
    {
        emit inSocketPosChanged();
        emit outSocketPosChanged();
    }
    else if (change == ItemZValueHasChanged)
    {
        int type = m_index.data(ROLE_NODETYPE).toInt();
        if ((type == BLACKBOARD_NODE || type == GROUP_NODE) && zValue() != ZVALUE_BLACKBOARD)
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

void ZenoNode::onCollaspeBtnClicked()
{
	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();

    STATUS_UPDATE_INFO info;
    info.role = ROLE_COLLASPED;
    info.newValue = !bCollasped;
    info.oldValue = bCollasped;
    pGraphsModel->updateNodeStatus(nodeId(), info, m_subGpIndex, true);
}

void ZenoNode::onOptionsBtnToggled(STATUS_BTN btn, bool toggled)
{
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());

	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	ZASSERT_EXIT(pGraphsModel);

    int options = m_index.data(ROLE_OPTIONS).toInt();
    int oldOpts = options;

    if (btn == STATUS_MUTE)
    {
        if (toggled)
        {
            options |= OPT_MUTE;
        }
        else
        {
            options ^= OPT_MUTE;
        }
    }
    else if (btn == STATUS_ONCE)
    {
        if (toggled)
        {
            options |= OPT_ONCE;
        }
        else
        {
            options ^= OPT_ONCE;
        }
    }
    else if (btn == STATUS_VIEW)
    {
		if (toggled)
		{
			options |= OPT_VIEW;
		}
		else
		{
			options ^= OPT_VIEW;
		}
    }
    else if (btn == STATUS_CACHE)
    {
        if (toggled)
        {
            options |= OPT_CACHE;
        }
        else
        {
            options ^= OPT_CACHE;
        }
    }

    STATUS_UPDATE_INFO info;
    info.role = ROLE_OPTIONS;
    info.newValue = options;
    info.oldValue = oldOpts;

    pGraphsModel->updateNodeStatus(nodeId(), info, m_subGpIndex, true);
}

void ZenoNode::onCollaspeUpdated(bool collasped)
{
    if (collasped)
    {
        m_bodyWidget->hide();
    }
    else
    {
        m_bodyWidget->show();
    }
    updateWhole();
    update();
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
    const QMimeData* pMimeData = QApplication::clipboard()->mimeData();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    const QString& str = scene()->property("link_label_info").toString();
    if (!str.isEmpty())
    {
        scene()->setProperty("link_label_info", "");
        pGraphsModel->beginTransaction(tr("add Link"));
        zeno::scope_exit sp([=]() { pGraphsModel->endTransaction(); });

        QModelIndex fromIndex = pGraphsModel->indexFromPath(str);
        if (fromIndex.isValid())
        {
            //remove the edge in inNode:inSock, if exists.
            int inProp = toIndex.data(ROLE_PARAM_SOCKPROP).toInt();
            if (inProp & SOCKPROP_DICTLIST_PANEL)
            {
                QString inSockType = toIndex.data(ROLE_PARAM_TYPE).toString();
                SOCKET_PROPERTY outProp = (SOCKET_PROPERTY)fromIndex.data(ROLE_PARAM_SOCKPROP).toInt();
                QString outSockType = fromIndex.data(ROLE_PARAM_TYPE).toString();
                QAbstractItemModel* pKeyObjModel =
                    QVariantPtr<QAbstractItemModel>::asPtr(toIndex.data(ROLE_VPARAM_LINK_MODEL));

                bool outSockIsContainer = false;
                if (inSockType == "list")
                {
                    outSockIsContainer = outSockType == "list";
                }
                else if (inSockType == "dict")
                {
                    const QModelIndex& fromNodeIdx = fromIndex.data(ROLE_NODE_IDX).toModelIndex();
                    const QString& outNodeCls = fromNodeIdx.data(ROLE_OBJNAME).toString();
                    const QString& outSockName = fromIndex.data(ROLE_PARAM_NAME).toString();
                    outSockIsContainer = outSockType == "dict" || (outNodeCls == "FuncBegin" && outSockName == "args");
                }

                //if outSock is a container, connects it as usual.
                if (outSockIsContainer)
                {
                    //legacy dict/list connection, and then we have to remove all inner dict key connection.
                    ZASSERT_EXIT(pKeyObjModel);
                    for (int r = 0; r < pKeyObjModel->rowCount(); r++)
                    {
                        const QModelIndex& keyIdx = pKeyObjModel->index(r, 0);
                        PARAM_LINKS links = keyIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                        for (QPersistentModelIndex _linkIdx : links)
                        {
                            pGraphsModel->removeLink(_linkIdx, true);
                        }
                    }
                }
                else
                {
                    //check multiple links
                    QModelIndexList fromSockets;
                    //check selected nodes.
                    //model: ViewParamModel
                    QString paramName = fromIndex.data(ROLE_PARAM_NAME).toString();
                    QString paramType = fromIndex.data(ROLE_PARAM_TYPE).toString();

                    QString toSockName = toIndex.data(ROLE_OBJPATH).toString();

                    // link to inner dict key automatically.
                    int n = pKeyObjModel->rowCount();
                    pGraphsModel->addExecuteCommand(new DictKeyAddRemCommand(true, pGraphsModel, toIndex.data(ROLE_OBJPATH).toString(), n));
                    toIndex = pKeyObjModel->index(n, 0);
                }
            }
            if (inProp != SOCKPROP_MULTILINK)
            {
                QPersistentModelIndex linkIdx;
                const PARAM_LINKS& links = toIndex.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
                if (!links.isEmpty())
                    linkIdx = links[0];
                if (linkIdx.isValid())
                    pGraphsModel->removeLink(linkIdx, true);
            }
            pGraphsModel->addLink(m_subGpIndex, fromIndex, toIndex, true);
        }
    }
}

void ZenoNode::onCustomNameChanged()
{
    m_NameItem->setTextInteractionFlags(Qt::NoTextInteraction);
    QString newText = m_NameItem->toPlainText();
    QString oldText = m_index.data(ROLE_CUSTOM_OBJNAME).toString();
    if (newText == oldText)
        return;
    QString name = m_index.data(ROLE_OBJNAME).toString();
    NODE_DESC desc;
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    pModel->getDescriptor(name, desc);
    QString category;
    if (!desc.categories.isEmpty())
        category = desc.categories[0];

    if (newText == name)
        newText = "";
    if (!pModel)
        return;
    if (!pModel->setCustomName(m_subGpIndex, m_index, newText))
    {
        newText = oldText.isEmpty() ? name : oldText;
        QMessageBox::warning(nullptr, tr("Warring"), tr("CustomName invalid!"));
    }

    if (newText.isEmpty() || newText == name)
    {
        m_NameItem->setText(name);
        m_pCategoryItem->setText(category);
        m_NameItem->setDefaultTextColor(QColor(255, 255, 255));
        ZGraphicsLayout::updateHierarchy(m_NameItem);
        ZGraphicsLayout::updateHierarchy(m_pCategoryItem);
    }
    else
    {
        QString text = category + ":" + name;
        if (text != m_pCategoryItem->text())
            m_pCategoryItem->setText(text);
        ZGraphicsLayout::updateHierarchy(m_pCategoryItem);
    }
}