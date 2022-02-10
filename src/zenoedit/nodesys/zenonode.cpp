#include "zenonode.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/model/subgraphmodel.h>
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenoparamnameitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include "zenoheatmapitem.h"
#include <zenoui/util/uihelper.h>


ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bInitSockets(false)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_collaspedWidget(nullptr)
    , m_bHeapMap(false)
    , m_pMainLayout(nullptr)
    , m_border(new QGraphicsRectItem)
    , m_NameItem(nullptr)
{
    setFlags(ItemIsMovable | ItemIsSelectable);
    setAcceptHoverEvents(true);
}

ZenoNode::~ZenoNode()
{
}

void ZenoNode::_initSocketItemPos()
{
    //need to optimizize
    QString nodeid = nodeId();
    for (auto sockName : m_inSockNames.keys())
    {
        auto sockLabelItem = m_inSockNames[sockName];
        auto socketItem = m_inSocks[sockName];
        QPointF scenePos = sockLabelItem->scenePos();
        QRectF sRect = sockLabelItem->sceneBoundingRect();
        QPointF pos = this->mapFromScene(scenePos);
        qreal x = -socketItem->size().width() / 2;
        qreal y = pos.y() + sRect.height() / 2 - socketItem->size().height() / 2;
        pos -= QPointF(m_renderParams.socketToText + socketItem->size().width(), 0);
        //fixed center on the border.
        //socket icon is hard to layout, as it's not a part of layout item but rely on the position of the layouted item.
        pos.setX(-socketItem->size().width() / 2 + m_renderParams.bodyBg.border_witdh / 2);
        pos.setY(y);

        socketItem->setPos(pos);
        emit socketPosInited(nodeid, sockName, true);
    }
    for (auto sockName : m_outSockNames.keys())
    {
        auto sockLabelItem = m_outSockNames[sockName];
        auto socketItem = m_outSocks[sockName];
        QRectF sRect = sockLabelItem->sceneBoundingRect();
        QPointF scenePos = sRect.topRight();
        sRect = mapRectFromScene(sRect);
        QPointF pos;

        int x = m_bodyWidget->rect().width() - socketItem->size().width() / 2 - m_renderParams.bodyBg.border_witdh / 2;
        int y = sRect.center().y() - socketItem->size().height() / 2;
        pos.setX(x);
        pos.setY(y);

        socketItem->setPos(pos);
        emit socketPosInited(nodeid, sockName, false);
    }
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
	QColor borderClr(250, 100, 0);
	borderClr.setAlphaF(0.2);
	qreal innerBdrWidth = 6;
	QPen pen(borderClr, 6);
	pen.setJoinStyle(Qt::MiterJoin);
	painter->setPen(pen);
	QRectF rc = m_pMainLayout->geometry();
	qreal offset = innerBdrWidth / 2; //finetune
	rc.adjust(-offset, -offset, offset, offset);
	QPainterPath path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
	painter->drawPath(path);

    //draw outter border
    qreal outterBdrWidth = 2;
    pen.setWidthF(outterBdrWidth);
    pen.setColor(QColor(250, 100, 0));
	painter->setPen(pen);
    offset = outterBdrWidth;
    rc.adjust(-offset, -offset, offset, offset);
    path = UiHelper::getRoundPath(rc, m_renderParams.headerBg.lt_radius, m_renderParams.headerBg.rt_radius, m_renderParams.bodyBg.lb_radius, m_renderParams.bodyBg.rb_radius, true);
    painter->drawPath(path);
}

QRectF ZenoNode::boundingRect() const
{
    return childrenBoundingRect();
}

int ZenoNode::type() const
{
    return Type;
}

void ZenoNode::initUI(const QModelIndex& index, SubGraphModel* pModel)
{
    if (true)
        initWangStyle(index, pModel);
    else
        initLegacy(index, pModel);

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();
}

void ZenoNode::initWangStyle(const QModelIndex& index, SubGraphModel* pModel)
{
    m_index = QPersistentModelIndex(index);
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_OBJTYPE).toInt());

    m_headerWidget = initHeaderWangStyle(type);
	m_bodyWidget = initBodyWidget(type);

	m_pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
	m_pMainLayout->addItem(m_headerWidget);
	m_pMainLayout->addItem(m_bodyWidget);
	m_pMainLayout->setContentsMargins(0, 0, 0, 0);
	m_pMainLayout->setSpacing(1);

    setLayout(m_pMainLayout);

	if (type == BLACKBOARD_NODE)
	{
		setZValue(ZVALUE_BLACKBOARD);
	}

	QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
	const QString& id = m_index.data(ROLE_OBJID).toString();
	setPos(pos);

	bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
	if (bCollasped)
		onCollaspeUpdated(true);

	// setPos will send geometry, but it's not supposed to happend during initialization.
	setFlag(ItemSendsGeometryChanges);
	setFlag(ItemSendsScenePositionChanges);

    m_headerWidget->installSceneEventFilter(this);

	connect(this, SIGNAL(doubleClicked(const QString&)), pModel, SLOT(onDoubleClicked(const QString&)));
}

void ZenoNode::initLegacy(const QModelIndex& index, SubGraphModel* pModel)
{
    m_index = QPersistentModelIndex(index);

    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_OBJTYPE).toInt());

    m_collaspedWidget = initCollaspedWidget();
    m_collaspedWidget->setVisible(false);

    m_headerWidget = initHeaderLegacy(type);
    if (type != BLACKBOARD_NODE) {
        initIndependentWidgetsLegacy();
    }
    m_bodyWidget = initBodyWidget(type);

    m_pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pMainLayout->addItem(m_collaspedWidget);
    m_pMainLayout->addItem(m_headerWidget);
    m_pMainLayout->addItem(m_bodyWidget);
    m_pMainLayout->setContentsMargins(0, 0, 0, 0);
    m_pMainLayout->setSpacing(0);

    setLayout(m_pMainLayout);

    if (type == BLACKBOARD_NODE)
    {
        setZValue(ZVALUE_BLACKBOARD);
    }

    QPointF pos = m_index.data(ROLE_OBJPOS).toPointF();
    const QString &id = m_index.data(ROLE_OBJID).toString();
    setPos(pos);

    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped)
        onCollaspeLegacyUpdated(true);

    // setPos will send geometry, but it's not supposed to happend during initialization.
    setFlag(ItemSendsGeometryChanges);
    setFlag(ItemSendsScenePositionChanges);

    connect(this, SIGNAL(doubleClicked(const QString&)), pModel, SLOT(onDoubleClicked(const QString&)));
}

void ZenoNode::initIndependentWidgetsLegacy()
{
    QRectF rc;

    rc = m_renderParams.rcMute;
    m_mute = new ZenoImageItem(m_renderParams.mute, QSizeF(rc.width(), rc.height()), this);
    m_mute->setPos(rc.topLeft());
    m_mute->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcView;
    m_view = new ZenoImageItem(m_renderParams.view, QSizeF(rc.width(), rc.height()), this);
    m_view->setPos(rc.topLeft());
    m_view->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcPrep;
    m_once = new ZenoImageItem(m_renderParams.prep, QSizeF(rc.width(), rc.height()), this);
    m_once->setPos(rc.topLeft());
    m_once->setZValue(ZVALUE_ELEMENT);

    rc = m_renderParams.rcCollasped;
    m_collaspe = new ZenoImageItem(m_renderParams.collaspe, QSizeF(rc.width(), rc.height()), this);
    m_collaspe->setPos(rc.topLeft());
    m_collaspe->setZValue(ZVALUE_ELEMENT);
    connect(m_collaspe, SIGNAL(clicked()), this, SLOT(onCollaspeBtnClicked()));
}

ZenoBackgroundWidget* ZenoNode::initCollaspedWidget()
{
    ZenoBackgroundWidget *widget = new ZenoBackgroundWidget(this);
    const auto &headerBg = m_renderParams.headerBg;
    widget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);

    QGraphicsLinearLayout *pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    const QString &name = m_index.data(ROLE_OBJNAME).toString();
    QFont font = m_renderParams.nameFont;
    font.setPointSize(font.pointSize() + 4);
    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(name, font, m_renderParams.nameClr.color());
    pNameItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    int horizontalPadding = 20;

    pHLayout->addItem(pNameItem);
    pHLayout->setAlignment(pNameItem, Qt::AlignLeft);
    pHLayout->addStretch();

    widget->setLayout(pHLayout);
    return widget;
}

ZenoBackgroundWidget *ZenoNode::initHeaderLegacy(NODE_TYPE type)
{
    ZenoBackgroundWidget* headerWidget = new ZenoBackgroundWidget(this);

    const auto &headerBg = m_renderParams.headerBg;
    headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
    headerWidget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);
    headerWidget->setBorder(headerBg.border_witdh, headerBg.clr_border);

    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    const QString &name = m_index.data(ROLE_OBJNAME).toString();
    ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(name, m_renderParams.nameFont, m_renderParams.nameClr.color());
    pNameItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);

    QFontMetrics metrics(m_renderParams.nameFont);
    int textWidth = metrics.horizontalAdvance(name);
    int horizontalPadding = 20;

    QRectF rc = m_renderParams.rcCollasped;

    ZenoSvgLayoutItem *collaspeItem = new ZenoSvgLayoutItem(m_renderParams.collaspe, QSizeF(rc.width(), rc.height()));
    collaspeItem->setZValue(ZVALUE_ELEMENT);
    collaspeItem->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    connect(collaspeItem, SIGNAL(clicked()), this, SLOT(onCollaspeBtnClicked()));

    pHLayout->addStretch();
    pHLayout->addItem(pNameItem);
    pHLayout->setAlignment(pNameItem, Qt::AlignCenter);
    pHLayout->addStretch();

    pHLayout->setContentsMargins(0, 5, 0, 5);
    pHLayout->setSpacing(5);

    headerWidget->setLayout(pHLayout);
    headerWidget->setZValue(ZVALUE_BACKGROUND);
    if (type == BLACKBOARD_NODE)
    {
        QColor clr(98, 108, 111);
        headerWidget->setColors(false, clr, clr, clr);
    }
    return headerWidget;
}

ZenoBackgroundWidget* ZenoNode::initHeaderWangStyle(NODE_TYPE type)
{
    ZenoBackgroundWidget* headerWidget = new ZenoBackgroundWidget(this);
	auto headerBg = m_renderParams.headerBg;
	headerWidget->setRadius(headerBg.lt_radius, headerBg.rt_radius, headerBg.lb_radius, headerBg.rb_radius);
	headerWidget->setColors(headerBg.bAcceptHovers, headerBg.clr_normal, headerBg.clr_hovered, headerBg.clr_selected);
	headerWidget->setBorder(headerBg.border_witdh, headerBg.clr_border);

    QGraphicsLinearLayout* pHLayout = new QGraphicsLinearLayout(Qt::Horizontal);

    ZenoSpacerItem* pSpacerItem = new ZenoSpacerItem(true, 100);

	const QString& name = m_index.data(ROLE_OBJNAME).toString();
	m_NameItem = new ZenoTextLayoutItem(name, m_renderParams.nameFont, m_renderParams.nameClr.color(), this);
	m_NameItem->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	QGraphicsLinearLayout* pNameLayout = new QGraphicsLinearLayout(Qt::Horizontal);
	pNameLayout->addItem(m_NameItem);
	pNameLayout->setContentsMargins(5, 5, 5, 5);

    m_pStatusWidgets = new ZenoMinStatusBtnWidget(m_renderParams.status, this);
    m_pStatusWidgets->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    int options = m_index.data(ROLE_OPTIONS).toInt();
    m_pStatusWidgets->setOptions(options);
    connect(m_pStatusWidgets, SIGNAL(toggleChanged(STATUS_BTN, bool)), this, SLOT(onOptionsBtnToggled(STATUS_BTN, bool)));

    pHLayout->addItem(pNameLayout);
    pHLayout->addItem(pSpacerItem);
    pHLayout->addItem(m_pStatusWidgets);
    pHLayout->setSpacing(0);
    pHLayout->setContentsMargins(0, 0, 0, 0);

	headerWidget->setLayout(pHLayout);
	headerWidget->setZValue(ZVALUE_BACKGROUND);
    headerWidget->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed));

	if (type == BLACKBOARD_NODE)
	{
		QColor clr(98, 108, 111);
		headerWidget->setColors(false, clr, clr, clr);
	}
	return headerWidget;
}

QSizeF ZenoNode::sizeHint(Qt::SizeHint which, const QSizeF& constraint) const
{
    if ("1d9489c4-zelloWorld" == nodeId())
    {
        int j;
        j = 0;
    }
    QSizeF sz = _base::sizeHint(which, constraint);
    return sz;
}

ZenoBackgroundWidget* ZenoNode::initBodyWidget(NODE_TYPE type)
{
    ZenoBackgroundWidget *bodyWidget = new ZenoBackgroundWidget(this);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal, bodyBg.clr_hovered, bodyBg.clr_selected);
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);
    pVLayout->setContentsMargins(0, 5, 0, 5);

    if (type != BLACKBOARD_NODE)
    {
        if (QGraphicsLayout* pParamsLayout = initParams())
        {
            pParamsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, 10, 10, 0);
            pVLayout->addItem(pParamsLayout);
        }
        if (QGraphicsGridLayout *pSocketsLayout = initSockets())
        {
            pSocketsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.paramsToTopSocket, m_renderParams.distParam.paramsLPadding, 0);
            pVLayout->addItem(pSocketsLayout);
        }

        //heapmap stays at the bottom of node layout.
        COLOR_RAMPS ramps = m_index.data(ROLE_COLORRAMPS).value<COLOR_RAMPS>();
        if (!ramps.isEmpty()) {
            ZenoHeatMapItem *pItem = new ZenoHeatMapItem(ramps);
            pVLayout->addItem(pItem);
        }
        bodyWidget->setZValue(ZVALUE_ELEMENT);
    }
    else
    {
        QColor clr(0, 0, 0);
        bodyWidget->setColors(false, clr, clr, clr);
        BLACKBOARD_INFO blackboard = m_index.data(ROLE_BLACKBOARD).value<BLACKBOARD_INFO>();

        ZenoBoardTextLayoutItem* pTextItem = new ZenoBoardTextLayoutItem(blackboard.content, m_renderParams.boardFont, m_renderParams.boardTextClr.color(), blackboard.sz);
        //pVLayout->addStretch();
        pVLayout->addItem(pTextItem);
    }

    bodyWidget->setLayout(pVLayout);
    return bodyWidget;
}

QGraphicsLayout* ZenoNode::initParams()
{
    const PARAMS_INFO &params = m_index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    QList<QString> names = params.keys();
    int r = 0, n = names.length();
    const QString nodeid = nodeId();

    QGraphicsLinearLayout* pParamsLayout = nullptr;
    if (n > 0)
    {
        pParamsLayout = new QGraphicsLinearLayout(Qt::Vertical);
        for (auto paramName : params.keys())
        {
            const PARAM_INFO &param = params[paramName];
            if (param.bEnableConnect)
                continue;

            QVariant val = param.value;
            QString value;
            if (val.type() == QVariant::String)
            {
                value = val.toString();
            }
            else if (val.type() == QVariant::Double)
            {
                value = QString::number(val.toDouble());
            }

            QGraphicsLinearLayout* pParamLayout = new QGraphicsLinearLayout(Qt::Horizontal);

            ZenoTextLayoutItem *pNameItem = new ZenoTextLayoutItem(paramName, m_renderParams.paramFont, m_renderParams.paramClr.color());
            pParamLayout->addItem(pNameItem);

            switch (param.control)
            {
                case CONTROL_STRING:
                case CONTROL_INT:
                case CONTROL_FLOAT:
                case CONTROL_BOOL:
                {
                    ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    pParamLayout->addItem(pLineEdit);
                    connect(pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
                        QString textValue = pLineEdit->text();
                        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                        SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
                        QVariant varValue;
                        switch (param.control) {
                        case CONTROL_INT: varValue = std::stoi(textValue.toStdString()); break;
                        case CONTROL_FLOAT: varValue = std::stof(textValue.toStdString()); break;
                        case CONTROL_BOOL: varValue = (bool)std::stoi(textValue.toStdString()); break;
                        case CONTROL_STRING: varValue = textValue; break;
                        }
                        pGraphModel->updateParam(nodeid, paramName, varValue, true);
                    });
                    m_paramControls[paramName] = pLineEdit;
                    break;
                }
                case CONTROL_ENUM:
                {
                    QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
                    ZenoParamComboBox* pComboBox = new ZenoParamComboBox(items, m_renderParams.comboboxParam);
                    pParamLayout->addItem(pComboBox);
                    connect(pComboBox, &ZenoParamComboBox::textActivated, this, [=](const QString& textValue) {
						QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
						SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
						pGraphModel->updateParam(nodeid, paramName, textValue, true);
                    });
                    m_paramControls[paramName] = pComboBox;
                    break;
                }
                case CONTROL_READPATH:
                {
                    ZenoParamLineEdit *pFileWidget = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    ZenoParamPushButton* pBtn = new ZenoParamPushButton("...");
                    pParamLayout->addItem(pFileWidget);
                    pParamLayout->addItem(pBtn);
                    connect(pFileWidget, &ZenoParamLineEdit::editingFinished, this, [=]() {
                        QString textValue = pFileWidget->text();
                        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                        SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
                        pGraphModel->updateParam(nodeid, paramName, textValue, true);
                    });
                    break;
                }
                case CONTROL_WRITEPATH:
                {
                    ZenoParamLineEdit *pFileWidget = new ZenoParamLineEdit(value, m_renderParams.lineEditParam);
                    ZenoParamPushButton *pBtn = new ZenoParamPushButton("...");
                    pParamLayout->addItem(pFileWidget);
                    pParamLayout->addItem(pBtn);
                    connect(pFileWidget, &ZenoParamLineEdit::editingFinished, this, [=]() {
                        QString textValue = pFileWidget->text();
                        QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
                        SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
                        pGraphModel->updateParam(nodeid, paramName, textValue, true);
                    });
                    break;
                }
                case CONTROL_MULTILINE_STRING:
                {
                    ZenoParamMultilineStr *pMultiStrEdit = new ZenoParamMultilineStr(value, m_renderParams.lineEditParam);
                    pParamLayout->addItem(pMultiStrEdit);
                    connect(pMultiStrEdit, &ZenoParamMultilineStr::editingFinished, this, [=]() {
                        QString textValue = pMultiStrEdit->text();
						QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
						SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
						pGraphModel->updateParam(nodeid, paramName, textValue, true);
                    });
                    m_paramControls[paramName] = pMultiStrEdit;
                    break;
                }
                case CONTROL_HEAPMAP:
                {
                    m_bHeapMap = true;
                    //break;
                }
                default:
                {
                    ZenoTextLayoutItem *pValueItem = new ZenoTextLayoutItem(value, m_renderParams.paramFont, m_renderParams.paramClr.color());
                    pParamLayout->addItem(pValueItem);
                    break;
                }
            }
            pParamsLayout->addItem(pParamLayout);
            r++;
        }
    }
    return pParamsLayout;
}

void ZenoNode::onParamUpdated(const QString &paramName, const QVariant &val)
{
    if (m_paramControls.find(paramName) != m_paramControls.end())
    {
        ZenoParamWidget* pWidget = m_paramControls[paramName];
        if (ZenoParamLineEdit* plineEdit = qobject_cast<ZenoParamLineEdit*>(pWidget))
        {
            plineEdit->setText(val.toString());
        }
        else if (ZenoParamComboBox* pComboBox = qobject_cast<ZenoParamComboBox*>(pWidget))
        {
            pComboBox->setText(val.toString());
        }
        else if (ZenoParamMultilineStr* pTextEdit = qobject_cast<ZenoParamMultilineStr*>(pWidget))
        {
            pTextEdit->setText(val.toString());
        }
    }
}

void ZenoNode::onNameUpdated(const QString& newName)
{
    m_NameItem->setPlainText(newName);
    update();
}

QGraphicsGridLayout* ZenoNode::initSockets()
{
    const QString &nodeid = nodeId();
    QGraphicsGridLayout *pSocketsLayout = new QGraphicsGridLayout;
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        int r = 0;
        for (auto inSock : inputs.keys()) {
            ZenoSocketItem *socket = new ZenoSocketItem(SOCKET_INFO(nodeid, inSock, QPointF(), true), m_renderParams.socket, m_renderParams.szSocket, this);
            m_inSocks.insert(std::make_pair(inSock, socket));
            socket->setZValue(ZVALUE_ELEMENT);

            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(inSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
            pSocketsLayout->addItem(pSocketItem, r, 0);
            m_inSockNames.insert(inSock, pSocketItem);

            r++;
        }
        OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        for (auto outSock : outputs.keys())
        {
            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(outSock, m_renderParams.socketFont, m_renderParams.socketClr.color());

            QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);
            pMiniLayout->addStretch();
            pMiniLayout->addItem(pSocketItem);
            pSocketsLayout->addItem(pMiniLayout, r, 1);

            m_outSockNames.insert(outSock, pSocketItem);

            ZenoSocketItem *socket = new ZenoSocketItem(SOCKET_INFO(nodeid, outSock, QPointF(), false), m_renderParams.socket, m_renderParams.szSocket, this);
            m_outSocks.insert(std::make_pair(outSock, socket));
            socket->setZValue(ZVALUE_ELEMENT);

            r++;
        }
    }
    return pSocketsLayout;
}

void ZenoNode::toggleSocket(bool bInput, const QString& sockName, bool bSelected)
{
    if (bInput) {
        auto itPort = m_inSocks.find(sockName);
        Q_ASSERT(itPort != m_inSocks.end());
        itPort->second->toggle(bSelected);
    } else {
        auto itPort = m_outSocks.find(sockName);
        Q_ASSERT(itPort != m_outSocks.end());
        itPort->second->toggle(bSelected);
    }
}

QPointF ZenoNode::getPortPos(bool bInput, const QString &portName)
{
    bool bCollasped = m_index.data(ROLE_COLLASPED).toBool();
    if (bCollasped)
    {
        QRectF rc = m_headerWidget->sceneBoundingRect();
        if (bInput)
        {
            return QPointF(rc.left(), rc.center().y());
        }
        else
        {
            return QPointF(rc.right(), rc.center().y());
        }
    }
    else
    {
        QString id = nodeId();
        if (bInput) {
            auto itPort = m_inSocks.find(portName);
            Q_ASSERT(itPort != m_inSocks.end());
            QPointF pos = itPort->second->sceneBoundingRect().center();
            return pos;
        } else {
            auto itPort = m_outSocks.find(portName);
            Q_ASSERT(itPort != m_outSocks.end());
            QPointF pos = itPort->second->sceneBoundingRect().center();
            return pos;
        }
    }
}

QString ZenoNode::nodeId() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJID).toString();
}

QString ZenoNode::nodeName() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJNAME).toString();
}

QPointF ZenoNode::nodePos() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OBJPOS).toPointF();
}

INPUT_SOCKETS ZenoNode::inputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
}

OUTPUT_SOCKETS ZenoNode::outputParams() const
{
    Q_ASSERT(m_index.isValid());
    return m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
}

bool ZenoNode::sceneEventFilter(QGraphicsItem* watched, QEvent* event)
{
    return _base::sceneEventFilter(watched, event);
}

bool ZenoNode::sceneEvent(QEvent *event)
{
    return _base::sceneEvent(event);
}

void ZenoNode::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event)
{
    _base::mouseDoubleClickEvent(event);
    QList<QGraphicsItem*> wtf = scene()->items(event->scenePos());
    if (wtf.contains(m_headerWidget))
    {
        onCollaspeBtnClicked();
    }
    else
    {
        emit doubleClicked(nodeName());
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
    _initSocketItemPos();
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
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
    SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
    Q_ASSERT(pGraphModel);

    bool bCollasped = pGraphModel->data(m_index, ROLE_COLLASPED).toBool();
    pGraphModel->setData(m_index, !bCollasped, ROLE_COLLASPED);
}

void ZenoNode::onOptionsBtnToggled(STATUS_BTN btn, bool toggled)
{
	QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
	SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel);
	Q_ASSERT(pGraphModel);

    int options = 0;
    if (btn == STATUS_MUTE)
    {
        if (toggled)
        {
            options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() | OPT_MUTE;
        }
        else
        {
            options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() ^ OPT_MUTE;
        }
        
    }
    else if (btn == STATUS_ONCE)
    {
        if (toggled)
        {
            options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() | OPT_ONCE;
        }
        else
        {
            options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() ^ OPT_ONCE;
        }
    }
    else if (btn == STATUS_VIEW)
    {
		if (toggled)
		{
			options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() | OPT_VIEW;
		}
		else
		{
			options = pGraphModel->data(m_index, ROLE_OPTIONS).toInt() ^ OPT_VIEW;
		}
    }

    pGraphModel->setData(m_index, options, ROLE_OPTIONS);
}

void ZenoNode::onCollaspeLegacyUpdated(bool collasped)
{
    if (collasped)
    {
        m_headerWidget->hide();
        m_bodyWidget->hide();
        for (auto p : m_inSocks) {
            p.second->hide();
        }
        for (auto p : m_outSocks) {
            p.second->hide();
        }
        m_mute->hide();
        m_view->hide();
        m_once->hide();

        m_collaspedWidget->show();
        m_collaspe->toggle(true);
    }
    else
    {
        m_bodyWidget->show();
        for (auto p : m_inSocks) {
            p.second->show();
        }
        for (auto p : m_outSocks) {
            p.second->show();
        }
        m_mute->show();
        m_view->show();
        m_once->show();
        m_headerWidget->show();
        m_collaspedWidget->hide();
        m_collaspe->toggle(false);
    }
    update();
}

void ZenoNode::onCollaspeUpdated(bool collasped)
{
    if (collasped)
    {
        m_bodyWidget->hide();
		for (auto p : m_inSocks) {
			p.second->hide();
		}
		for (auto p : m_outSocks) {
			p.second->hide();
		}
        m_pMainLayout->setSpacing(0);
    }
    else
    {
		m_bodyWidget->show();
		for (auto p : m_inSocks) {
			p.second->show();
		}
		for (auto p : m_outSocks) {
			p.second->show();
		}
        m_pMainLayout->setSpacing(1);
    }
    update();
}

void ZenoNode::onOptionsUpdated(int options)
{
    m_pStatusWidgets->setOptions(options);
}

QVariant ZenoNode::itemChange(GraphicsItemChange change, const QVariant &value)
{
    if (change == QGraphicsItem::ItemSelectedHasChanged)
    {
        bool bSelected = isSelected();
        m_headerWidget->toggle(bSelected);
        m_bodyWidget->toggle(bSelected);
    }
    else if (change == QGraphicsItem::ItemPositionChange)
    {
        emit nodePositionChange(nodeId());
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        QPointF pos = this->scenePos();
        int x = pos.x(), y = pos.y();
        x = x - x % SCENE_GRID_SIZE;
        y = y - y % SCENE_GRID_SIZE;
        if (x != pos.x() && y != pos.y())
        {
            pos.setX(x);
            pos.setY(y);
            QAbstractItemModel* pModel = const_cast<QAbstractItemModel*>(m_index.model());
            if (SubGraphModel* pGraphModel = qobject_cast<SubGraphModel*>(pModel))
            {
                pGraphModel->setData(pGraphModel->index(nodeId()), pos, ROLE_OBJPOS);
            }
        }
    }
    return value;
}
