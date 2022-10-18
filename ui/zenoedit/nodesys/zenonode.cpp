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

    //params.
    const PARAMS_INFO& params = m_index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    QList<QString> names = params.keys();
    int n = names.length();
    const QString nodeid = nodeId();

    for (auto paramName : params.keys())
    {
        const PARAM_INFO& param = params[paramName];
        if (param.bEnableConnect)
            continue;

        ZGraphicsLayout* pParamLayout = initParam(param.control, paramName, param, pScene);
        if (pParamLayout)
        {
            m_bodyLayout->addLayout(pParamLayout);
        }
    }

    ZGraphicsLayout* pCustomParams = initCustomParamWidgets();
    if (pCustomParams)
    {
        m_bodyLayout->addLayout(pCustomParams);
    }

    onSocketsUpdate(pScene, true, true);
    onSocketsUpdate(pScene, false, true);

    bodyWidget->setLayout(m_bodyLayout);

    return bodyWidget;
}

ZGraphicsLayout* ZenoNode::initCustomParamWidgets()
{
    return nullptr;
}

ZGraphicsLayout* ZenoNode::initParam(PARAM_CONTROL ctrl, const QString& paramName, const PARAM_INFO& param, ZenoSubGraphScene* pScene)
{
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
    paramCtrl.ctrl_layout->addItem(paramCtrl.param_name, Qt::AlignVCenter);

	switch (param.control)
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
            ZenoParamWidget* pWidget = initParamWidget(pScene, param);
            paramCtrl.ctrl_layout->addItem(pWidget);
            paramCtrl.param_control = pWidget;
            break;
	    }
	    default:
	    {
		    //zeno::log_warn("got undefined control type {}", param.control);
            const QString& value = UiHelper::variantToString(param.value);
            QGraphicsTextItem* pValueItem = new QGraphicsTextItem(value);
            pValueItem->setFont(m_renderParams.paramFont);
            pValueItem->setDefaultTextColor(m_renderParams.paramClr.color());
            paramCtrl.ctrl_layout->addItem(pValueItem);
		    break;
	    }
	}
    //todo: mark this as const func.
    m_params.insert(paramName, paramCtrl);

    return paramCtrl.ctrl_layout;
}

ZenoParamWidget* ZenoNode::initParamWidget(ZenoSubGraphScene* scene, const PARAM_INFO& param)
{
    const PARAM_CONTROL ctrl = param.control;
    const QString& paramName = param.name;
    if (param.control == CONTROL_NONVISIBLE)
        return nullptr;

    Callback_EditFinished cbUpdateParam = [=](QVariant newValue) {
        onParamEditFinished(paramName, newValue);
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    const QVariant& deflValue = param.value;
    ZenoParamWidget* pControl = zenoui::createItemWidget(deflValue, ctrl, param.typeDesc, cbUpdateParam, scene, cbSwith);
    return pControl;
}

QPersistentModelIndex ZenoNode::subGraphIndex() const
{
    return m_subGpIndex;
}

void ZenoNode::onParamEditFinished(const QString& paramName, const QVariant& value)
{
    // graphics item sync to model.
    const QString nodeid = nodeId();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

    PARAM_UPDATE_INFO info;

    const PARAMS_INFO& params = m_index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    ZASSERT_EXIT(params.find(paramName) != params.end());
    const PARAM_INFO& param = params[paramName];

    info.oldValue = param.value;
    info.name = paramName;

    switch (param.control)
    {
    case CONTROL_COLOR:
    case CONTROL_CURVE:
    case CONTROL_VEC:
        info.newValue = value;
        break;
    default:
        info.newValue = UiHelper::parseTextValue(param.control, value.toString());
        break;
    }
    if (info.oldValue != info.newValue)
        pGraphsModel->updateParamInfo(nodeid, info, m_subGpIndex, true);
}

void ZenoNode::onParamUpdated(ZenoSubGraphScene* pScene, const QString& paramName, const QVariant& val)
{
    auto createNewWidget = [this](ZenoParamWidget* pWidget,
        ZGraphicsLayout* pParamLayout,
        ZenoSubGraphScene* pScene,
        const PARAM_INFO& param)->ZenoParamWidget*
    {
        pParamLayout->removeItem(pWidget);
        delete pWidget;
        ZenoParamWidget* pNewWidget = initParamWidget(pScene, param);
        ZASSERT_EXIT(pNewWidget, nullptr);
        pParamLayout->addItem(pNewWidget);
        m_params[param.name].param_control = pNewWidget;
        return pNewWidget;
    };

    if (m_params.find(paramName) != m_params.end())
    {
        ZenoParamWidget* pWidget = m_params[paramName].param_control;
        ZGraphicsLayout* pParamLayout = m_params[paramName].ctrl_layout;

        PARAMS_INFO params = index().data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        const PARAM_INFO& param = params[paramName];
        bool bUpdateLayout = false;

        QString oldType = pWidget ? pWidget->data(GVKEY_TYPE).toString() : "";
        if (oldType != param.typeDesc /*|| param.control != gv->controlType()*/)
        {
            pParamLayout->removeItem(pWidget);
            pWidget = createNewWidget(pWidget, pParamLayout, pScene, param);
            pParamLayout->addItem(pWidget);
        }
        //set value on pControl.
        ZenoGvHelper::setValue(pWidget, param.control, param.value);
        updateWhole();
    }
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

bool ZenoNode::renameDictKey(bool bInput, const INPUT_SOCKETS& inputs, const OUTPUT_SOCKETS& outputs)
{
    if (bInput)
    {
        QStringList coreKeys = inputs.keys();
        QStringList uiKeys = m_inSockets.keys();
        QSet<QString> coreNewSet = coreKeys.toSet().subtract(uiKeys.toSet());
        QSet<QString> uiNewSet = uiKeys.toSet().subtract(coreKeys.toSet());
        //detecting the rename case for "MakeDict".
        if (coreNewSet.size() == 1 && uiNewSet.size() == 1 && m_index.data(ROLE_OBJNAME) == "MakeDict")
        {
            //rename.
            QString oldName = *uiNewSet.begin();
            QString newName = *coreNewSet.begin();

            auto it = m_inSockets.find(oldName);
            ZASSERT_EXIT(oldName != newName && it != m_inSockets.end() && it->second, false);
            it->second->updateSockName(newName);
            it->first = newName;

            ZGraphicsLayout::updateHierarchy(it->second);
            return true;
        }
        return false;
    }
    else
    {
        QStringList coreKeys = outputs.keys();
        QStringList uiKeys = m_outSockets.keys();
        QSet<QString> coreNewSet = coreKeys.toSet().subtract(uiKeys.toSet());
        QSet<QString> uiNewSet = uiKeys.toSet().subtract(coreKeys.toSet());
        //detecting the rename case for "MakeDict".
        if (coreNewSet.size() == 1 && uiNewSet.size() == 1 && m_index.data(ROLE_OBJNAME) == "ExtractDict")
        {
            //rename.
            QString oldName = *uiNewSet.begin();
            QString newName = *coreNewSet.begin();

            auto it = m_outSockets.find(oldName);
            ZASSERT_EXIT(oldName != newName && it != m_outSockets.end() && it->second, false);
            it->second->updateSockName(newName);
            it->first = newName;

            ZGraphicsLayout::updateHierarchy(it->second);
            return true;
        }
        return false;
    }
}

void ZenoNode::onSocketsUpdate(ZenoSubGraphScene* pScene, bool bInput, bool bInit)
{
    const QString &nodeid = nodeId();
    const QString& nodeName = m_index.data(ROLE_OBJNAME).toString();

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    auto cbFuncRenameSock = [=](QString oldText, QString newText) {
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);
        SOCKET_UPDATE_INFO info;
        info.bInput = bInput;
        info.newInfo.name = newText;
        info.oldInfo.name = oldText;
        info.updateWay = SOCKET_UPDATE_NAME;
        bool ret = pGraphsModel->updateSocketNameNotDesc(m_index.data(ROLE_OBJID).toString(), info, m_subGpIndex, true);
    };

    Callback_OnSockClicked cbSockOnClick = [=](ZenoSocketItem* pSocketItem) {
        emit socketClicked(pSocketItem);
    };

    if (bInput)
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        if (renameDictKey(true, inputs, OUTPUT_SOCKETS()))
        {
            updateWhole();
            return;
        }
        for (const INPUT_SOCKET inSocket : inputs)
        {
            const QString& inSock = inSocket.info.name;
            if (m_inSockets.find(inSock) == m_inSockets.end())
            {
                //add socket
                bool bEditableSock = inSocket.info.control == CONTROL_DICTKEY;
                ZSocketLayout* pMiniLayout = new ZSocketLayout(m_index, inSock, true, bEditableSock, cbSockOnClick, cbFuncRenameSock);

                const QString &sockType = inSocket.info.type;
                PARAM_CONTROL ctrl = inSocket.info.control;

                ZenoParamWidget* pSocketControl = initSocketWidget(pScene, inSocket);
                pMiniLayout->setControl(pSocketControl);
                if (pSocketControl)
                    pSocketControl->setVisible(inSocket.linkIndice.isEmpty());

                m_inSockets.insert(inSock, pMiniLayout);

                if (!bInit && pModel->IsSubGraphNode(m_index))
                {
                    //dynamic socket added, ensure that the key is above the SRC key.
                    m_bodyLayout->insertLayout(0, pMiniLayout);
                }
                else
                {
                    m_bodyLayout->addLayout(pMiniLayout);
                }
            }
            else
            {
                //when socket type changed
                ZSocketLayout* pControlLayout = m_inSockets[inSocket.info.name];
                ZASSERT_EXIT(pControlLayout);

                QGraphicsItem* pControl = pControlLayout->control();
                QString oldType = pControl ? pControl->data(GVKEY_TYPE).toString() : "";
                if (oldType != inSocket.info.type /*|| inSocket.info.control != gv->controlType()*/)
                {
                    QGraphicsItem* pNewControl = initSocketWidget(pScene, inSocket);
                    pControlLayout->setControl(pNewControl);
                    pControl = pNewControl;
                    updateWhole();
                }
                //set value on pControl.
                ZenoGvHelper::setValue(pControl, inSocket.info.control, inSocket.info.defaultValue);
            }
        }
        //remove all keys which don't exist in model.
        for (auto name : m_inSockets.keys())
        {
            if (inputs.find(name) == inputs.end())
            {
                ZSocketLayout* pSocketLayout = m_inSockets[name];
                ZGraphicsLayout* pParentLayout = pSocketLayout->parentLayout();
                m_inSockets.remove(name);
                pParentLayout->removeLayout(pSocketLayout);
            }
        }
    }
    else
    {
        OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        if (renameDictKey(false, INPUT_SOCKETS(), outputs))
        {
            return;
        }
        for (OUTPUT_SOCKET outSocket : outputs)
        {
            const QString& outSock = outSocket.info.name;
            if (m_outSockets.find(outSock) == m_outSockets.end())
            {
                bool bEditableSock = outSocket.info.control == CONTROL_DICTKEY;
                ZSocketLayout* pSockLayout = new ZSocketLayout(m_index, outSock, false, bEditableSock, cbSockOnClick, cbFuncRenameSock);

                m_outSockets[outSock] = pSockLayout;

                if (!bInit && pModel->IsSubGraphNode(m_index))
                {
                    //dynamic socket added, ensure that the key is above the DST key.
                    m_bodyLayout->insertLayout(0, pSockLayout);
                }
                else
                {
                    m_bodyLayout->addLayout(pSockLayout);
                }
            }
        }

        //remove all keys which don't exist in model.
        for (auto name : m_outSockets.keys())
        {
            if (outputs.find(name) == outputs.end())
            {
                ZSocketLayout* pSockLayout = m_outSockets[name];
                if (pSockLayout)
                {
                    ZGraphicsLayout* parent = pSockLayout->parentLayout();
                    if (parent)
                    {
                        parent->removeLayout(parent);
                        m_outSockets.remove(name);
                        ZGraphicsLayout::updateHierarchy(parent);
                    }
                }
            }
        }
    }
    if (!bInit)
        updateWhole();
}

ZenoParamWidget* ZenoNode::initSocketWidget(ZenoSubGraphScene* scene, const INPUT_SOCKET inSocket)
{
    const QString& nodeid = nodeId();
    const QString& inSock = inSocket.info.name;
    const QString& sockType = inSocket.info.type;

    auto cbUpdateSocketDefl = [=](QVariant state) {
        bool bOk = false;
        INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
        ZASSERT_EXIT(bOk);
        updateSocketDeflValue(nodeid, inSock, _inSocket, state);
    };

    auto cbSwith = [=](bool bOn) {
        zenoApp->getMainWindow()->setInDlgEventLoop(bOn);
    };

    PARAM_CONTROL ctrl = inSocket.info.control;
    const QVariant& deflValue = inSocket.info.defaultValue;
    ZenoParamWidget* pControl = zenoui::createItemWidget(deflValue, ctrl, sockType, cbUpdateSocketDefl, scene, cbSwith);
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

void ZenoNode::updateSocketDeflValue(const QString& nodeid, const QString& inSock, const INPUT_SOCKET& inSocket, const QVariant& newValue)
{
    INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();

    PARAM_UPDATE_INFO info;
    info.name = inSock;
    info.oldValue = inputs[inSock].info.defaultValue;
    info.newValue = newValue;

    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
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
            INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (!inputs[name].linkIndice.isEmpty())
                linkIdx = inputs[name].linkIndice[0];
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
            OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
            if (!outputs[name].linkIndice.isEmpty())
                linkIdx = outputs[name].linkIndice[0];
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
