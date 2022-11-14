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
#include <zenoui/comctrl/dialog/curvemap/zcurvemapeditor.h>
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
#include <zenoui/comctrl/gv/zgraphicstextitem.h>
#include <zenoui/comctrl/gv/zenogvhelper.h>
#include <zenomodel/include/iparammodel.h>
#include <zenomodel/include/viewparammodel.h>


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_border(new QGraphicsRectItem)
    , m_NameItem(nullptr)
    , m_bError(false)
    , m_bEnableSnap(false)
    , m_bodyLayout(nullptr)
    , m_bUIInited(false)
    , m_viewParams(nullptr)
    , m_inputsLayout(nullptr)
    , m_paramsLayout(nullptr)
    , m_outputsLayout(nullptr)
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
}

void ZenoNode::_drawBorderWangStyle(QPainter* painter)
{
	//draw inner border
	painter->setRenderHint(QPainter::Antialiasing, true);
    QColor baseColor = /*m_bError ? QColor(200, 84, 79) : */QColor(255, 100, 0);
	QColor borderClr(baseColor);
	borderClr.setAlphaF(0.2);
	qreal innerBdrWidth = 6;
	QPen pen(borderClr, 6);
	pen.setJoinStyle(Qt::MiterJoin);
	painter->setPen(pen);

	QRectF rc = boundingRect();
	qreal offset = innerBdrWidth / 2; //finetune
	rc.adjust(-offset, -offset, offset, offset);
	QPainterPath path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
	painter->drawPath(path);

    //draw outter border
    qreal outterBdrWidth = 2;
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

    m_headerWidget = initHeaderWidget();
    m_bodyWidget = initBodyWidget(pScene);

    ZGraphicsLayout* mainLayout = new ZGraphicsLayout(false);
    mainLayout->addItem(m_headerWidget);
    mainLayout->addItem(m_bodyWidget);
    mainLayout->setSpacing(0);
    setLayout(mainLayout);

    if (type == BLACKBOARD_NODE) {
        setZValue(ZVALUE_BLACKBOARD);
    }

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

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();

    m_bUIInited = true;
}

ZLayoutBackground* ZenoNode::initHeaderWidget()
{
    ZLayoutBackground* headerWidget = new ZLayoutBackground;
    auto headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);

    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel && m_index.isValid(), nullptr);

    QColor clrHeaderBg;
    if (pGraphsModel->IsSubGraphNode(m_index))
        clrHeaderBg = QColor(86, 143, 131);
    else
        clrHeaderBg = headerBg.clr_normal;

    headerWidget->setColors(headerBg.bAcceptHovers, clrHeaderBg, clrHeaderBg, clrHeaderBg);
    headerWidget->setBorder(headerBg.border_witdh, headerBg.clr_border);

    const QString& name = m_index.data(ROLE_OBJNAME).toString();
    m_NameItem = new ZSimpleTextItem(name);
    m_NameItem->setBrush(QColor(226, 226, 226));
    QFont font2("HarmonyOS Sans Bold", 14);
    font2.setBold(true);
    m_NameItem->setFont(font2);
    m_NameItem->updateBoundingRect();

    m_pStatusWidgets = new ZenoMinStatusBtnWidget(m_renderParams.status);
    m_pStatusWidgets->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    int options = m_index.data(ROLE_OPTIONS).toInt();
    m_pStatusWidgets->setOptions(options);
    connect(m_pStatusWidgets, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    ZGraphicsLayout* pHLayout = new ZGraphicsLayout(true);
    pHLayout->addSpacing(10);
    pHLayout->addItem(m_NameItem, Qt::AlignVCenter);
    pHLayout->addSpacing(100);
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
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    m_bodyLayout = new ZGraphicsLayout(false);
    m_bodyLayout->setSpacing(5);
    m_bodyLayout->setContentsMargin(16, 16, 16, 16);

    m_viewParams = QVariantPtr<ViewParamModel>::asPtr(m_index.data(ROLE_NODEPARAMS));
    QStandardItem* root = m_viewParams->invisibleRootItem();
    ZASSERT_EXIT(root && root->rowCount() == 3, nullptr);
    //see ViewParamModel::initNode()
    QStandardItem* inputsItem = root->child(0);
    QStandardItem* paramsItem = root->child(1);
    QStandardItem* outputsItem = root->child(2);

    connect(m_viewParams, &ViewParamModel::rowsInserted, this, &ZenoNode::onViewParamInserted);
    connect(m_viewParams, &ViewParamModel::rowsAboutToBeRemoved, this, &ZenoNode::onViewParamAboutToBeRemoved);
    connect(m_viewParams, &ViewParamModel::dataChanged, this, &ZenoNode::onViewParamDataChanged);

    //params.
    m_paramsLayout = initParams(paramsItem, pScene);
    m_bodyLayout->addLayout(m_paramsLayout);

    m_inputsLayout = initSockets(inputsItem, true, pScene);
    m_bodyLayout->addLayout(m_inputsLayout);

    m_outputsLayout = initSockets(outputsItem, false, pScene);
    m_bodyLayout->addLayout(m_outputsLayout);

    bodyWidget->setLayout(m_bodyLayout);
    return bodyWidget;
}

ZGraphicsLayout* ZenoNode::initCustomParamWidgets()
{
    return nullptr;
}

ZenoParamWidget* ZenoNode::initParamWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx)
{
    const PARAM_CONTROL ctrl = (PARAM_CONTROL)paramIdx.data(ROLE_PARAM_CTRL).toInt();
    if (ctrl == CONTROL_NONVISIBLE)
        return nullptr;

    const QPersistentModelIndex perIdx(paramIdx);

    Callback_EditFinished cbUpdateParam = [=](QVariant newValue) {
        //sync param from view to model.
        const QString& paramName = perIdx.data(ROLE_PARAM_NAME).toString();
        const PARAM_CONTROL ctrl = (PARAM_CONTROL)perIdx.data(ROLE_PARAM_CTRL).toInt();
        const QString nodeid = nodeId();
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

        PARAM_UPDATE_INFO info;
        info.oldValue = perIdx.data(ROLE_PARAM_VALUE);
        info.name = paramName;

        switch (ctrl)
        {
        case CONTROL_COLOR:
        case CONTROL_CURVE:
        case CONTROL_VEC:
            info.newValue = newValue;
            break;
        default:
            info.newValue = UiHelper::parseTextValue(ctrl, newValue.toString());
            break;
        }
        if (info.oldValue != info.newValue)
            pGraphsModel->updateParamInfo(nodeid, info, m_subGpIndex, true);
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    const QVariant& deflValue = paramIdx.data(ROLE_PARAM_VALUE);
    const QString& typeDesc = paramIdx.data(ROLE_PARAM_TYPE).toString();
    ZenoParamWidget* pControl = zenoui::createItemWidget(deflValue, ctrl, typeDesc, cbUpdateParam, scene, cbSwith);
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
        m_NameItem->setText(newName);
        ZGraphicsLayout::updateHierarchy(m_NameItem);
    }
}

void ZenoNode::onViewParamDataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight, const QVector<int>& roles)
{
    if (roles.isEmpty())
        return;

    int role = roles[0];
    QStandardItem* pItem = m_viewParams->itemFromIndex(topLeft);
    ZASSERT_EXIT(pItem);
    int vType = pItem->data(ROLE_VPARAM_TYPE).toInt();
    if (vType == VPARAM_GROUP)
    {

    }
    if (vType != VPARAM_PARAM)
    {
        return;
    }

    QModelIndex viewParamIdx = pItem->index();

    QStandardItem* parentItem = pItem->parent();
    ZASSERT_EXIT(parentItem);
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    const QString& groupName = parentItem->text();
    const QString& paramName = pItem->data(ROLE_VPARAM_NAME).toString();

    if (role == ROLE_VPARAM_NAME)
    {
        if (groupName == "In Sockets")
        {
            for (auto it = m_inSockets.begin(); it != m_inSockets.end(); it++)
            {
                if (it->second->viewSocketIdx() == viewParamIdx)
                {
                    QString oldName = it->first;
                    it->first = paramName;
                    break;
                }
            }
        }
        else if (groupName == "Parameters")
        {
            for (auto it = m_params.begin(); it != m_params.end(); it++)
            {
                if (it->second.viewidx == viewParamIdx)
                {
                    QString oldName = it->first;
                    it->first = paramName;
                    break;
                }
            }
        }
        else if (groupName == "Out Sockets")
        {
            for (auto it = m_outSockets.begin(); it != m_outSockets.end(); it++)
            {
                if (it->second->viewSocketIdx() == viewParamIdx)
                {
                    QString oldName = it->first;
                    it->first = paramName;
                    break;
                }
            }
        }
        return;
    }

    if (groupName == "In Sockets")
    {
        ZASSERT_EXIT(m_inSockets.find(paramName) != m_inSockets.end());
        ZSocketLayout* pControlLayout = pControlLayout = m_inSockets[paramName];
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
                    QGraphicsItem* pNewControl = initSocketWidget(pScene, parentItem->index());
                    pControlLayout->setControl(pNewControl);
                    pControl = pNewControl;
                    updateWhole();
                }
                //set value on pControl.
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, ctrl, deflValue);
                break;
            }
            case ROLE_PARAM_VALUE:
            {
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(pControl, ctrl, deflValue);
                break;
            }
        }
    }
    else if (groupName == "Parameters")
    {
        const QString& sockName = viewParamIdx.data(ROLE_PARAM_NAME).toString();
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

                ZenoParamWidget* pNewControl = initParamWidget(pScene, viewParamIdx);
                if (pNewControl)
                {
                    pParamLayout->addItem(pNewControl);
                    m_params[sockName].param_control = pNewControl;
                }
                updateWhole();
                break;
            }
            case ROLE_PARAM_VALUE:
            {
                const QVariant& deflValue = pItem->data(ROLE_PARAM_VALUE);
                ZenoGvHelper::setValue(paramCtrl.param_control, ctrl, deflValue);
                break;
            }
        }
    }
}

void ZenoNode::onViewParamInserted(const QModelIndex& parent, int first, int last)
{
    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    if (!parent.isValid())
    {
        QStandardItem* root = m_viewParams->invisibleRootItem();
        ZASSERT_EXIT(root);
        QStandardItem* pGroup = root->child(first);
        if (!pGroup) return;

        const QString& groupName = pGroup->text();
        if (groupName == "Parameters")
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
        else if (groupName == "In Sockets")
        {
            ZGraphicsLayout* playout = initSockets(pGroup, true, pScene);
            if (m_inputsLayout)
            {
                m_inputsLayout->clear();
                m_bodyLayout->removeLayout(m_inputsLayout);
            }
            m_inputsLayout = playout;
            m_bodyLayout->insertLayout(1, m_inputsLayout);
            updateWhole();
        }
        else if (groupName == "Out Sockets")
        {
            ZGraphicsLayout* playout = initSockets(pGroup, false, pScene);
            if (m_outputsLayout)
            {
                m_outputsLayout->clear();
                m_bodyLayout->removeLayout(m_outputsLayout);
            }
            m_outputsLayout = playout;
            m_bodyLayout->insertLayout(2, m_outputsLayout);
            updateWhole();
        }
        return;
    }

    QStandardItem* parentItem = m_viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    QStandardItem* paramItem = parentItem->child(first);
    const QString& groupName = parentItem->text();
    const QModelIndex& viewParamIdx = paramItem->index();

    //see ViewParamModel::initNode
    if (groupName == "In Sockets" || groupName == "Out Sockets")
    {
        bool bInput = groupName == "In Sockets";
        ZGraphicsLayout* pSocketsLayout = bInput ? m_inputsLayout : m_outputsLayout;
        ZSocketLayout* pSocketLayout = addSocket(viewParamIdx, bInput, pScene);
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        if (pModel->IsSubGraphNode(m_index))
        {
            //dynamic socket added, ensure that the key is above the SRC key.
            pSocketsLayout->insertLayout(0, pSocketLayout);
        }
        else
        {
            pSocketsLayout->addLayout(pSocketLayout);
        }
        updateWhole();
    }
    else if (groupName == "Parameters")
    {
        m_paramsLayout->addLayout(addParam(viewParamIdx, pScene));
    }
}

void ZenoNode::onViewParamAboutToBeRemoved(const QModelIndex& parent, int first, int last)
{
    if (!parent.isValid())
    {
        ZASSERT_EXIT(first == 0 && last == 2);  //corresponding to ViewParamModel::clone.
        //remove all component.
        m_paramsLayout->clear();
        m_inputsLayout->clear();
        m_outputsLayout->clear();
        return;
    }

    ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(this->scene());
    ZASSERT_EXIT(pScene);

    QStandardItem* parentItem = m_viewParams->itemFromIndex(parent);
    ZASSERT_EXIT(parentItem->data(ROLE_VPARAM_TYPE) == VPARAM_GROUP);
    QStandardItem* paramItem = parentItem->child(first);
    const QString& groupName = parentItem->text();
    const QModelIndex& viewParamIdx = paramItem->index();

    if (groupName == "In Sockets" || groupName == "Out Sockets")
    {
        bool bInput = groupName == "In Sockets";
        const QString& sockName = viewParamIdx.data(ROLE_VPARAM_NAME).toString();
        ZSocketLayout* pSocketLayout = bInput ? m_inSockets[sockName] : m_outSockets[sockName];
        if (bInput)
            m_inSockets.remove(sockName);
        else
            m_outSockets.remove(sockName);
        ZGraphicsLayout* pParentLayout = pSocketLayout->parentLayout();
        pParentLayout->removeLayout(pSocketLayout);
    }
    else if (groupName == "Parameters")
    {
        const QString& paramName = viewParamIdx.data(ROLE_PARAM_NAME).toString();

        ZASSERT_EXIT(m_params.find(paramName) != m_params.end());
        ZGraphicsLayout* paramLayout = m_params[paramName].ctrl_layout;
        m_params.remove(paramName);
        ZGraphicsLayout* pParentLayout = paramLayout->parentLayout();
        if (pParentLayout)
            pParentLayout->removeLayout(paramLayout);
    }
}

ZGraphicsLayout* ZenoNode::initSockets(QStandardItem* socketItems, const bool bInput, ZenoSubGraphScene* pScene)
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel && m_index.isValid(), nullptr);

    IParamModel* socketModel = pGraphsModel->paramModel(m_index, bInput ? PARAM_INPUT : PARAM_OUTPUT);
    ZASSERT_EXIT(socketModel, nullptr);

    ZGraphicsLayout* pSocketsLayout = new ZGraphicsLayout(false);
    pSocketsLayout->setSpacing(5);

    for (int r = 0; r < socketItems->rowCount(); r++)
    {
        const QStandardItem* pItem = socketItems->child(r);
        const QModelIndex& viewIdx = pItem->index();
        ZSocketLayout* pSocketLayout = addSocket(viewIdx, bInput, pScene);
        pSocketsLayout->addLayout(pSocketLayout);
    }
    return pSocketsLayout;
}

ZSocketLayout* ZenoNode::addSocket(const QModelIndex& viewSockIdx, bool bInput, ZenoSubGraphScene* pScene)
{
    auto cbFuncRenameSock = [=](QString oldText, QString newText) {
        //todo:
        ZASSERT_EXIT(false);

        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);
        IParamModel* pModel = pGraphsModel->paramModel(m_index, bInput ? PARAM_INPUT : PARAM_OUTPUT);
        ZASSERT_EXIT(pModel);
        bool ret = pModel->setData(viewSockIdx, newText, ROLE_PARAM_NAME);
    };

    Callback_OnSockClicked cbSockOnClick = [=](ZenoSocketItem* pSocketItem) {
        emit socketClicked(pSocketItem);
    };

    const QString& sockName = viewSockIdx.data(ROLE_VPARAM_NAME).toString();
    PARAM_CONTROL ctrl = (PARAM_CONTROL)viewSockIdx.data(ROLE_PARAM_CTRL).toInt();
    const QString& sockType = viewSockIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& deflVal = viewSockIdx.data(ROLE_PARAM_VALUE);
    const PARAM_LINKS& links = viewSockIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();

    bool bEditableSock = ctrl == CONTROL_DICTKEY;
    ZSocketLayout* pMiniLayout = new ZSocketLayout(viewSockIdx, sockName, bInput, bEditableSock, cbSockOnClick, cbFuncRenameSock);

    if (bInput)
    {
        ZenoParamWidget* pSocketControl = initSocketWidget(pScene, viewSockIdx);
        pMiniLayout->setControl(pSocketControl);
        if (pSocketControl)
            pSocketControl->setVisible(links.isEmpty());
    }

    if (bInput)
        m_inSockets[sockName] = pMiniLayout;
    else
        m_outSockets[sockName] = pMiniLayout;

    return pMiniLayout;
}

ZGraphicsLayout* ZenoNode::initParams(QStandardItem* paramItems, ZenoSubGraphScene* pScene)
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel && m_index.isValid(), nullptr);

    IParamModel* paramsModel = pGraphsModel->paramModel(m_index, PARAM_PARAM);
    ZASSERT_EXIT(paramsModel, nullptr);

    ZGraphicsLayout* paramsLayout = new ZGraphicsLayout(false);
    paramsLayout->setSpacing(5);

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
    paramCtrl.param_name = textItem;
    paramCtrl.viewidx = viewparamIdx;
    paramCtrl.ctrl_layout->addItem(paramCtrl.param_name, Qt::AlignVCenter);

    switch (ctrl)
    {
        case CONTROL_STRING:
        case CONTROL_INT:
        case CONTROL_FLOAT:
        case CONTROL_BOOL:
        case CONTROL_VEC:
        case CONTROL_ENUM:
        case CONTROL_READPATH:
        case CONTROL_WRITEPATH:
        case CONTROL_MULTILINE_STRING:
        case CONTROL_CURVE:
        {
            ZenoParamWidget* pWidget = initParamWidget(pScene, viewparamIdx);
            paramCtrl.ctrl_layout->addItem(pWidget);
            paramCtrl.param_control = pWidget;
            break;
        }
        default:
        {
            break;
        }
    }

    m_params.insert(paramName, paramCtrl);
    return paramCtrl.ctrl_layout;
}

ZenoParamWidget* ZenoNode::initSocketWidget(ZenoSubGraphScene* scene, const QModelIndex& paramIdx)
{
    const QPersistentModelIndex perIdx(paramIdx);

    auto cbUpdateSocketDefl = [=](QVariant newValue) {
        if (!perIdx.isValid())
            return;
        QAbstractItemModel* paramModel = const_cast<QAbstractItemModel*>(perIdx.model());
        paramModel->setData(perIdx, newValue, ROLE_PARAM_VALUE);
        return;

        /* old style*/
        bool bOk = false;
        const QString& nodeid = nodeId();

        PARAM_UPDATE_INFO info;
        info.name = perIdx.data(ROLE_PARAM_NAME).toString();
        info.oldValue = perIdx.data(ROLE_PARAM_VALUE);
        info.newValue = newValue;

        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);

        if (newValue.type() == QMetaType::VoidStar)
        {
            //curvemodel: 
            //todo: only store as a void ptr, have to develope a undo/redo mechasim for "submodel".
            pGraphsModel->updateSocketDefl(nodeid, info, m_subGpIndex, false);
        }
        else if (info.oldValue != info.newValue)
        {
            pGraphsModel->updateSocketDefl(nodeid, info, m_subGpIndex, true);
        }
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    PARAM_CONTROL ctrl = (PARAM_CONTROL)paramIdx.data(ROLE_PARAM_CTRL).toInt();
    const QString& sockType = paramIdx.data(ROLE_PARAM_TYPE).toString();
    const QVariant& deflVal = paramIdx.data(ROLE_PARAM_VALUE);

    ZenoParamWidget* pControl = zenoui::createItemWidget(deflVal, ctrl, sockType, cbUpdateSocketDefl, scene, cbSwith);
    return pControl;
}

void ZenoNode::onSocketLinkChanged(const QString& sockName, bool bInput, bool bAdded)
{
	if (bInput)
	{
        if (m_inSockets.find(sockName) != m_inSockets.end())
        {
            ZenoSocketItem* pSocket = m_inSockets[sockName]->socketItem();
            pSocket->toggle(bAdded);
            pSocket->setSockStatus(bAdded ? ZenoSocketItem::STATUS_CONNECTED : ZenoSocketItem::STATUS_NOCONN);
            if (m_inSockets[sockName]->control())
            {
                m_inSockets[sockName]->control()->setVisible(!bAdded);
                updateWhole();
            }
        }
	}
	else
	{
        if (m_outSockets.find(sockName) != m_outSockets.end())
        {
            ZenoSocketItem* pSocket = m_outSockets[sockName]->socketItem();
            pSocket->toggle(bAdded);
            pSocket->setSockStatus(bAdded ? ZenoSocketItem::STATUS_CONNECTED : ZenoSocketItem::STATUS_NOCONN);
        }
	}
}

void ZenoNode::getSocketInfoByItem(ZenoSocketItem* pSocketItem, QString& sockName, QPointF& scenePos, bool& bInput, QPersistentModelIndex& linkIdx)
{
    for (auto name : m_inSockets.keys())
    {
        auto socketLayout = m_inSockets[name];
        if (socketLayout->socketItem() == pSocketItem)
        {
            bInput = true;
            sockName = name;
            scenePos = pSocketItem->center();
            const QModelIndex& paramIdx = pSocketItem->paramIndex();
            const PARAM_LINKS& links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
                linkIdx = links[0];
            return;
        }
    }
    for (auto name : m_outSockets.keys())
    {
        auto socketLayout = m_outSockets[name];
        if (socketLayout->socketItem() == pSocketItem)
        {
            bInput = false;
            sockName = name;
            scenePos = pSocketItem->center();
            const QModelIndex& paramIdx = pSocketItem->paramIndex();
            const PARAM_LINKS& links = paramIdx.data(ROLE_PARAM_LINKS).value<PARAM_LINKS>();
            if (!links.isEmpty())
                linkIdx = links[0];
            return;
        }
    }
}

void ZenoNode::toggleSocket(bool bInput, const QString& sockName, bool bSelected)
{
    if (bInput) {
        ZASSERT_EXIT(m_inSockets.find(sockName) != m_inSockets.end());
        ZenoSocketItem* pSocket = m_inSockets[sockName]->socketItem();
        pSocket->toggle(bSelected);
    } else {
        ZASSERT_EXIT(m_outSockets.find(sockName) != m_outSockets.end());
        ZenoSocketItem* pSocket = m_outSockets[sockName]->socketItem();
        pSocket->toggle(bSelected);
    }
}

void ZenoNode::markError(bool isError)
{
    m_bError = isError;
    ZASSERT_EXIT(m_headerWidget);
    if (m_bError)
        m_headerWidget->setColors(false, QColor(200, 84, 79), QColor(), QColor());
    else
        m_headerWidget->setColors(false, QColor(83, 96, 147), QColor(), QColor());
    update();
}

ZenoSocketItem* ZenoNode::getSocketItem(bool bInput, const QString& sockName)
{
    if (bInput) {
        ZASSERT_EXIT(m_inSockets.find(sockName) != m_inSockets.end(), nullptr);
        return m_inSockets[sockName]->socketItem();
    } else {
        ZASSERT_EXIT(m_outSockets.find(sockName) != m_outSockets.end(), nullptr);
        return m_outSockets[sockName]->socketItem();
    }
}

ZenoSocketItem* ZenoNode::getNearestSocket(const QPointF& pos, bool bInput)
{
    ZenoSocketItem* pItem = nullptr;
    float minDist = std::numeric_limits<float>::max();
    auto socks = bInput ? m_inSockets : m_outSockets;
    for (ZSocketLayout* sock : socks)
    {
        //todo: socket now is a children of sockettext.
        QPointF sockPos = sock->socketItem()->center();
        QPointF offset = sockPos - pos;
        float dist = std::sqrt(offset.x() * offset.x() + offset.y() * offset.y());
        if (dist < minDist)
        {
            minDist = dist;
            pItem = sock->socketItem();
        }
    }
    return pItem;
}

QPointF ZenoNode::getPortPos(bool bInput, const QString &portName)
{
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped) {
        QRectF rc = m_headerWidget->sceneBoundingRect();
        if (bInput) {
            return QPointF(rc.left(), rc.center().y());
        } else {
            return QPointF(rc.right(), rc.center().y());
        }
    } else {
        QString id = nodeId();
        if (bInput) {
            ZASSERT_EXIT(m_inSockets.find(portName) != m_inSockets.end(), QPointF());
            QPointF pos = m_inSockets[portName]->socketItem()->center();
            return pos;
        } else {
            ZASSERT_EXIT(m_outSockets.find(portName) != m_outSockets.end(), QPointF());
            QPointF pos = m_outSockets[portName]->socketItem()->center();
            return pos;
        }
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

INPUT_SOCKETS ZenoNode::inputParams() const
{
    ZASSERT_EXIT(m_index.isValid(), INPUT_SOCKETS());
    return m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
}

OUTPUT_SOCKETS ZenoNode::outputParams() const
{
    ZASSERT_EXIT(m_index.isValid(), OUTPUT_SOCKETS());
    return m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
}

void ZenoNode::onUpdateParamsNotDesc()
{
}

bool ZenoNode::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
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
    if (pGraphsModel && pGraphsModel->IsSubGraphNode(m_index))
    {
        scene()->clearSelection();
        this->setSelected(true);

        QMenu *nodeMenu = new QMenu;
        QAction *pCopy = new QAction("Copy");
        QAction *pPaste = new QAction("Paste");
        QAction *pDelete = new QAction("Delete");

        nodeMenu->addAction(pCopy);
        nodeMenu->addAction(pPaste);
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
        nodeMenu->exec(QCursor::pos());
        nodeMenu->deleteLater();
    }
    else
    {
        NODE_CATES cates = zenoApp->graphsManagment()->currentModel()->getCates();
        QPointF pos = event->scenePos();
        ZenoNewnodeMenu *menu = new ZenoNewnodeMenu(m_subGpIndex, cates, pos);
        menu->setEditorFocus();
        menu->exec(pos.toPoint());
        menu->deleteLater();
    }
}

void ZenoNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
    QList<QGraphicsItem*> wtf = scene()->items(event->scenePos());
    if (wtf.contains(m_headerWidget))
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
    }
}

void ZenoNode::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseMoveEvent(event);
}

void ZenoNode::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseReleaseEvent(event);
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
    m_pStatusWidgets->blockSignals(true);
    m_pStatusWidgets->setOptions(options);
    m_pStatusWidgets->blockSignals(false);
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

        ZenoMainWindow* mainWin = zenoApp->getMainWindow();
        mainWin->onNodesSelected(m_subGpIndex, { index() }, bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        if (m_bEnableSnap)
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
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        QPointF newPos = value.toPointF();
        QPointF oldPos = m_index.data(ROLE_OBJPOS).toPointF();
        if (newPos != oldPos)
        {
            STATUS_UPDATE_INFO info;
            info.role = ROLE_OBJPOS;
            info.newValue = newPos;
            info.oldValue = oldPos;
            pGraphsModel->updateNodeStatus(nodeId(), info, m_subGpIndex, false);

            bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
            QRectF rc = m_headerWidget->sceneBoundingRect();

            emit inSocketPosChanged();
            emit outSocketPosChanged();
        }
    }
    return value;
}

void ZenoNode::updateWhole()
{
    ZGraphicsLayout::updateHierarchy(this);
    emit inSocketPosChanged();
    emit outSocketPosChanged();
}
