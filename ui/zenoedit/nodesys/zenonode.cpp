#include "zenonode.h"
#include <zenoui/model/modelrole.h>
#include "model/subgraphmodel.h"
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenoparamnameitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/include/igraphsmodel.h>
#include <zeno/utils/logger.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zveceditor.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "graphsmanagment.h"
#include "../nodesview/zenographseditor.h"
#include "util/log.h"


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
    , m_mute(nullptr)
    , m_view(nullptr)
    , m_once(nullptr)
    , m_collaspe(nullptr)
    , m_bError(false)
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
    for (auto sockName : m_inSockets.keys())
    {
        auto sockLabelItem = m_inSockets[sockName].socket_text;
        auto socketItem = m_inSockets[sockName].socket;
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
    for (auto sockName : m_outSockets.keys())
    {
        auto sockLabelItem = m_outSockets[sockName].socket_text;
        auto socketItem = m_outSockets[sockName].socket;
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
    QColor baseColor = /*m_bError ? QColor(200, 84, 79) : */QColor(255, 100, 0);
	QColor borderClr(baseColor);
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
    pen.setColor(baseColor);
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

void ZenoNode::initUI(const QModelIndex& subGIdx, const QModelIndex& index)
{
    if (true)
        initWangStyle(subGIdx, index);
    else
        initLegacy(subGIdx, index);

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();
}

void ZenoNode::initWangStyle(const QModelIndex& subGIdx, const QModelIndex& index)
{
    m_index = QPersistentModelIndex(index);
    m_subGpIndex = QPersistentModelIndex(subGIdx);
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

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

    //todo:
	//connect(this, SIGNAL(doubleClicked(const QString&)), pModel, SLOT(onDoubleClicked(const QString&)));
}

void ZenoNode::initLegacy(const QModelIndex& subGIdx, const QModelIndex& index)
{
    m_index = QPersistentModelIndex(index);

    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

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

    //connect(this, SIGNAL(doubleClicked(const QString&)), pModel, SLOT(onDoubleClicked(const QString&)));
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
        if (QGraphicsLayout* pSocketsLayout = initSockets())
        {
            pSocketsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.paramsToTopSocket, m_renderParams.distParam.paramsLPadding, 0);
            pVLayout->addItem(pSocketsLayout);
        }
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

            QGraphicsLinearLayout* pParamLayout = new QGraphicsLinearLayout(Qt::Horizontal);
            initParam(param.control, pParamLayout, paramName, param);
            pParamsLayout->addItem(pParamLayout);
            r++;
        }
    }
    QGraphicsLinearLayout* pCustomParams = initCustomParamWidgets();
    if (pParamsLayout && pCustomParams)
        pParamsLayout->addItem(pCustomParams);
    return pParamsLayout;
}

QGraphicsLinearLayout* ZenoNode::initCustomParamWidgets()
{
    return nullptr;
}

void ZenoNode::initParam(PARAM_CONTROL ctrl, QGraphicsLinearLayout* pParamLayout, const QString& paramName, const PARAM_INFO& param)
{
    const QString& value = UiHelper::variantToString(param.value);

	ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem(paramName, m_renderParams.paramFont, m_renderParams.paramClr.color());
	pParamLayout->addItem(pNameItem);

	switch (param.control)
	{
	    case CONTROL_STRING:
	    case CONTROL_INT:
	    case CONTROL_FLOAT:
	    case CONTROL_BOOL:
	    {
		    ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(value, param.control,  m_renderParams.lineEditParam);
		    pParamLayout->addItem(pLineEdit);
		    connect(pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pLineEdit->text());
			    });
		    m_paramControls[paramName] = pLineEdit;
		    break;
	    }
        case CONTROL_VEC3F:
        {
            QVector<qreal> vec = param.value.value<QVector<qreal>>();
            ZenoVecEditWidget* pVecEditor = new ZenoVecEditWidget(vec);
            pParamLayout->addItem(pVecEditor);
			connect(pVecEditor, &ZenoVecEditWidget::editingFinished, this, [=]() {
				
				});
			m_paramControls[paramName] = pVecEditor;
            break;
        }
	    case CONTROL_ENUM:
	    {
		    QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
		    ZenoParamComboBox* pComboBox = new ZenoParamComboBox(items, m_renderParams.comboboxParam);
		    pParamLayout->addItem(pComboBox);

		    connect(pComboBox, &ZenoParamComboBox::textActivated, this, [=](const QString& textValue) {
                onParamEditFinished(param.control, paramName, textValue);
			    });
		    m_paramControls[paramName] = pComboBox;
		    break;
	    }
	    case CONTROL_READPATH:
	    {
		    ZenoParamLineEdit* pFileWidget = new ZenoParamLineEdit(value, param.control, m_renderParams.lineEditParam);

			ImageElement elem;
			elem.image = ":/icons/ic_openfile.svg";
            elem.imageHovered = ":/icons/ic_openfile-on.svg";
			elem.imageOn = ":/icons/ic_openfile-on.svg";
            ZenoSvgLayoutItem* openBtn = new ZenoSvgLayoutItem(elem, QSizeF(30, 30));

		    pParamLayout->addItem(pFileWidget);
		    pParamLayout->addItem(openBtn);
			pParamLayout->setItemSpacing(1, 0);
			pParamLayout->setItemSpacing(2, 0);

		    connect(pFileWidget, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pFileWidget->text());
			});
            connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                QString path = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
                if (path.isEmpty())
                    return;
                pFileWidget->setText(path);
                onParamEditFinished(param.control, paramName, path);
            });
            m_paramControls[paramName] = pFileWidget;
		    break;
	    }
	    case CONTROL_WRITEPATH:
	    {
		    ZenoParamLineEdit* pFileWidget = new ZenoParamLineEdit(value, param.control, m_renderParams.lineEditParam);

            ImageElement elem;
			elem.image = ":/icons/ic_openfile.svg";
            elem.imageHovered = ":/icons/ic_openfile-on.svg";
			elem.imageOn = ":/icons/ic_openfile-on.svg";

            ZenoSvgLayoutItem* openBtn = new ZenoSvgLayoutItem(elem, QSizeF(30, 30));

		    pParamLayout->addItem(pFileWidget);
		    pParamLayout->addItem(openBtn);
            pParamLayout->setItemSpacing(1, 0);
            pParamLayout->setItemSpacing(2, 0);
		    connect(pFileWidget, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pFileWidget->text());
			});
            connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                QString path = QFileDialog::getSaveFileName(nullptr, "Path to Save", "", "All Files(*);;");
                if (path.isEmpty())
                    return;
                pFileWidget->setText(path);
                onParamEditFinished(param.control, paramName, path);
            });
            m_paramControls[paramName] = pFileWidget;
		    break;
	    }
	    case CONTROL_MULTILINE_STRING:
	    {
		    ZenoParamMultilineStr* pMultiStrEdit = new ZenoParamMultilineStr(value, m_renderParams.lineEditParam);
		    pParamLayout->addItem(pMultiStrEdit);
		    connect(pMultiStrEdit, &ZenoParamMultilineStr::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pMultiStrEdit->text());
			    });
		    m_paramControls[paramName] = pMultiStrEdit;
		    break;
	    }
	    case CONTROL_HEATMAP:
	    case CONTROL_CURVE:
	    {
		    ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(value, param.control,  m_renderParams.lineEditParam);
		    pParamLayout->addItem(pLineEdit);
		    connect(pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pLineEdit->text());
			    });
		    m_paramControls[paramName] = pLineEdit;
		    break;
	    }
	    default:
	    {
		    zeno::log_warn("got undefined control type {}", param.control);
		    ZenoTextLayoutItem* pValueItem = new ZenoTextLayoutItem(value, m_renderParams.paramFont, m_renderParams.paramClr.color());
		    pParamLayout->addItem(pValueItem);
		    break;
	    }
	}
}

QPersistentModelIndex ZenoNode::subGraphIndex() const
{
    return m_subGpIndex;
}

void ZenoNode::onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue)
{
    // graphics item sync to model.

    const QString nodeid = nodeId();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

    PARAM_UPDATE_INFO info;
    info.oldValue = pGraphsModel->getParamValue(nodeid, paramName, m_subGpIndex);
    info.newValue = UiHelper::parseTextValue(editCtrl, textValue);
    info.name = paramName;
    if (info.oldValue != info.newValue)
        pGraphsModel->updateParamInfo(nodeid, info, m_subGpIndex, true);
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
        else if (ZenoVecEditWidget* pVecEdit = qobject_cast<ZenoVecEditWidget*>(pWidget))
        {
            pVecEdit->setVec(val.value<QVector<qreal>>());
        }
    }
}

void ZenoNode::onSocketDeflUpdated(const PARAM_UPDATE_INFO& info)
{
    if (m_inSockets.find(info.name) != m_inSockets.end())
    {
        ZenoParamWidget* pWidget = m_inSockets[info.name].socket_control;
        if (ZenoParamLineEdit* plineEdit = qobject_cast<ZenoParamLineEdit*>(pWidget))
        {
            plineEdit->setText(info.newValue.toString());
        }
        else if (ZenoVecEditWidget* pVecEdit = qobject_cast<ZenoVecEditWidget*>(pWidget))
        {
            pVecEdit->setVec(info.newValue.value<QVector<qreal>>());
        }
    }
}

void ZenoNode::onInOutSocketChanged(bool bInput)
{
    if (!m_index.isValid())
        return;

    if (bInput)
    {
        const INPUT_SOCKETS& inSocks = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        for (const INPUT_SOCKET& inSocket : inSocks)
        {
            switch (inSocket.info.control)
            {
			    case CONTROL_STRING:
			    case CONTROL_INT:
			    case CONTROL_FLOAT:
			    case CONTROL_BOOL:
                {
                    ZenoParamLineEdit* plineEdit = qobject_cast<ZenoParamLineEdit*>(m_inSockets[inSocket.info.name].socket_control);
                    if (plineEdit)
                    {
                        plineEdit->setText(inSocket.info.defaultValue.toString());
                    }
                    break;
                }
                case CONTROL_VEC3F:
                {
                    ZenoVecEditWidget* pVecEdit = qobject_cast<ZenoVecEditWidget*>(m_inSockets[inSocket.info.name].socket_control);
                    if (pVecEdit)
                    {
                        const QVector<qreal>& vec = inSocket.info.defaultValue.value<QVector<qreal>>();
                        pVecEdit->setVec(vec);
                    }
                    break;
                }
                case CONTROL_ENUM:
                {
                    ZenoParamComboBox* pComboBox = qobject_cast<ZenoParamComboBox*>(m_inSockets[inSocket.info.name].socket_control);
                    if (pComboBox)
                    {
                        pComboBox->setText(inSocket.info.defaultValue.toString());
                    }
                    break;
                }
            }
        }
    }
    else
    {
        const OUTPUT_SOCKETS& outSocks = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        for (const OUTPUT_SOCKET& outSocket : outSocks)
        {
            //todo
        }
    }
}

void ZenoNode::onSocketUpdated(const SOCKET_UPDATE_INFO& info)
{
    switch (info.updateWay)
    {
        case SOCKET_INSERT:
        {
            const QString& newName = info.newInfo.name;
			if (info.bInput)
			{
                ZASSERT_EXIT(m_inSockets.find(newName) == m_inSockets.end());

                _socket_ctrl sock;
                sock.socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
                sock.socket_text = new ZenoTextLayoutItem(newName, m_renderParams.socketFont, m_renderParams.socketClr.color());
                m_inSockets[newName] = sock;

                int r = std::max(m_inSockets.count() - 1, 0);
                m_pSocketsLayout->insertItem(r, sock.socket_text);
                m_pSocketsLayout->updateGeometry();
			}
			else
			{
				ZASSERT_EXIT(m_outSockets.find(newName) == m_outSockets.end());

                _socket_ctrl sock;
				sock.socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
				sock.socket_text = new ZenoTextLayoutItem(newName, m_renderParams.socketFont, m_renderParams.socketClr.color());
                sock.socket_text->setRight(true);
                m_outSockets[newName] = sock;

                int r = std::max(m_inSockets.count() + m_outSockets.count() - 1, 0);
                m_pSocketsLayout->insertItem(r, sock.socket_text);
                m_pSocketsLayout->updateGeometry();
			}
            break;
        }
        case SOCKET_REMOVE:
        {
            const QString& name = info.oldInfo.name;
            if (info.bInput)
            {
                ZASSERT_EXIT(m_inSockets.find(name) != m_inSockets.end());
                const _socket_ctrl& sock = m_inSockets[name];
                m_pSocketsLayout->removeItem(sock.socket_text);
                delete sock.socket;
                delete sock.socket_text;
                m_inSockets.remove(name);
                m_pSocketsLayout->updateGeometry();
            }
            else
            {
                ZASSERT_EXIT(m_outSockets.find(name) != m_outSockets.end());
				const _socket_ctrl& sock = m_outSockets[name];
				m_pSocketsLayout->removeItem(sock.socket_text);
				delete sock.socket;
				delete sock.socket_text;
                m_outSockets.remove(name);
				m_pSocketsLayout->updateGeometry();
            }
            break;
        }
        case SOCKET_UPDATE_NAME:
        {
		    const QString& oldName = info.oldInfo.name;
		    const QString& newName = info.newInfo.name;
		    if (info.bInput)
		    {
			    ZASSERT_EXIT(m_inSockets.find(oldName) != m_inSockets.end());
			    m_inSockets[newName] = m_inSockets[oldName];
			    m_inSockets[newName].socket_text->setPlainText(newName);
			    m_inSockets.remove(oldName);
		    }
		    else
		    {
			    ZASSERT_EXIT(m_outSockets.find(oldName) != m_outSockets.end());
			    m_outSockets[newName] = m_outSockets[oldName];
			    m_outSockets[newName].socket_text->setPlainText(info.newInfo.name);
			    m_outSockets.remove(oldName);
		    }
            break;
        }
    }
}

void ZenoNode::onNameUpdated(const QString& newName)
{
    ZASSERT_EXIT(m_NameItem);
    if (m_NameItem)
    {
		m_NameItem->setPlainText(newName);
		m_NameItem->updateGeometry();
		if (auto layoutItem = m_NameItem->parentLayoutItem())
			layoutItem->updateGeometry();
	}
}

void ZenoNode::onSocketLinkChanged(const QString& sockName, bool bInput, bool bAdded)
{
	if (bInput)
	{
        ZASSERT_EXIT(m_inSockets.find(sockName) != m_inSockets.end());
        m_inSockets[sockName].socket->toggle(bAdded);
	}
	else
	{
		ZASSERT_EXIT(m_outSockets.find(sockName) != m_outSockets.end());
        m_outSockets[sockName].socket->toggle(bAdded);
	}
}

void ZenoNode::updateSocketDeflValue(const QString& nodeid, const QString& inSock, const INPUT_SOCKET& inSocket, const QVariant& newValue)
{
    INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();

    PARAM_UPDATE_INFO info;
    info.name = inSock;
    info.oldValue = inputs[inSock].info.defaultValue;
    info.newValue = newValue;

    if (info.oldValue != info.newValue)
    {
        IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pGraphsModel);
        pGraphsModel->updateSocketDefl(nodeid, info, m_subGpIndex, true);
    }
}

QGraphicsLayout* ZenoNode::initSockets()
{
    m_pSocketsLayout = new QGraphicsLinearLayout(Qt::Vertical);

    const QString &nodeid = nodeId();
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
        int r = 0;
        for (auto inSock : inputs.keys())
        {
            ZenoSocketItem *socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
            socket->setZValue(ZVALUE_ELEMENT);

            QGraphicsLinearLayout* pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);

            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(inSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
            pMiniLayout->addItem(pSocketItem);

            m_pSocketsLayout->addItem(pMiniLayout);

            _socket_ctrl socket_ctrl;

            const INPUT_SOCKET& inSocket = inputs[inSock];
            //dont need lineedit except "int£¬float£¬vec3f£¬string".
            const QString& sockType = inSocket.info.type;
            PARAM_CONTROL ctrl = inSocket.info.control;

            if (sockType == "int" || sockType == "float" || sockType == "string" || sockType == "readpath" || sockType == "writepath")
            {
				ZenoParamLineEdit* pSocketEditor = new ZenoParamLineEdit(UiHelper::variantToString(inSocket.info.defaultValue), inSocket.info.control, m_renderParams.lineEditParam);
                pMiniLayout->addItem(pSocketEditor);
                //todo: allow to edit path directly?
                connect(pSocketEditor, &ZenoParamLineEdit::editingFinished, this, [=]()
                {
                    const QVariant &newValue = UiHelper::_parseDefaultValue(pSocketEditor->text(), inSocket.info.type);
                    updateSocketDeflValue(nodeid, inSock, inSocket, newValue);
				});
                socket_ctrl.socket_control = pSocketEditor;

                if (sockType == "readpath" || sockType == "writepath")
                {
                    ImageElement elem;
                    elem.image = ":/icons/ic_openfile.svg";
                    elem.imageHovered = ":/icons/ic_openfile-on.svg";
                    elem.imageOn = ":/icons/ic_openfile-on.svg";

                    bool isRead = sockType == "readpath";

                    ZenoSvgLayoutItem *openBtn = new ZenoSvgLayoutItem(elem, QSizeF(30, 30));
                    connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                        QString path;
                        if (isRead) {
                            path = QFileDialog::getOpenFileName(nullptr, "File to Open", "", "All Files(*);;");
                        }
                        else {
                            path = QFileDialog::getSaveFileName(nullptr, "Path to Save", "", "All Files(*);;");
                        }
                        if (path.isEmpty())
                            return;
                        pSocketEditor->setText(path);
                        updateSocketDeflValue(nodeid, inSock, inSocket, path);
                    });

                    pMiniLayout->addItem(openBtn);
                    pMiniLayout->setItemSpacing(1, 0);
                    pMiniLayout->setItemSpacing(2, 0);
                }
            }
            else if (sockType == "vec3f")
            {
                QVector<qreal> vec = inSocket.info.defaultValue.value<QVector<qreal>>();
				ZenoVecEditWidget* pVecEditor = new ZenoVecEditWidget(vec);
                pMiniLayout->addItem(pVecEditor);
				connect(pVecEditor, &ZenoVecEditWidget::editingFinished, this, [=]()
                {
                    QVector<qreal> vec = pVecEditor->vec();
                    const QVariant& newValue = QVariant::fromValue(vec);
                    updateSocketDeflValue(nodeid, inSock, inSocket, newValue);
				});
                socket_ctrl.socket_control = pVecEditor;
            }
            else if (ctrl == CONTROL_ENUM)
            {
                QStringList items = sockType.mid(QString("enum ").length()).split(QRegExp("\\s+"));
                ZenoParamComboBox *pComboBox = new ZenoParamComboBox(items, m_renderParams.comboboxParam);
                pMiniLayout->addItem(pComboBox);
                QString val = inSocket.info.defaultValue.toString();
                if (items.indexOf(val) != -1) {
                    pComboBox->setText(val);
                }

                connect(pComboBox, &ZenoParamComboBox::textActivated, this,
                        [=](const QString &textValue)
                {
                    QString oldValue = pComboBox->text();
                    updateSocketDeflValue(nodeid, inSock, inSocket, textValue);
                });
                socket_ctrl.socket_control = pComboBox;
            }

            socket_ctrl.socket = socket;
            socket_ctrl.socket_text = pSocketItem;
            m_inSockets.insert(inSock, socket_ctrl);

            r++;
        }
        OUTPUT_SOCKETS outputs = m_index.data(ROLE_OUTPUTS).value<OUTPUT_SOCKETS>();
        for (auto outSock : outputs.keys())
        {
            ZenoTextLayoutItem *pSocketItem = new ZenoTextLayoutItem(outSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
            pSocketItem->setRight(true);

            QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);
            pMiniLayout->addStretch();
            pMiniLayout->addItem(pSocketItem);
            m_pSocketsLayout->addItem(pMiniLayout);

            ZenoSocketItem *socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
            socket->setZValue(ZVALUE_ELEMENT);

            _socket_ctrl socket_ctrl;
            socket_ctrl.socket = socket;
            socket_ctrl.socket_text = pSocketItem;
            m_outSockets[outSock] = socket_ctrl;

            r++;
        }
    }
    return m_pSocketsLayout;
}

void ZenoNode::getSocketInfoByItem(ZenoSocketItem* pSocketItem, QString& sockName, QPointF& scenePos, bool& bInput, QPersistentModelIndex& linkIdx)
{
    for (auto name : m_inSockets.keys())
    {
        auto ctrl = m_inSockets[name];
        if (ctrl.socket == pSocketItem)
        {
            bInput = true;
            sockName = name;
            scenePos = pSocketItem->sceneBoundingRect().center();
            INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
            if (!inputs[name].linkIndice.isEmpty())
                linkIdx = inputs[name].linkIndice[0];
            return;
        }
    }
    for (auto name : m_outSockets.keys())
    {
        auto ctrl = m_outSockets[name];
        if (ctrl.socket == pSocketItem)
        {
            bInput = false;
            sockName = name;
            scenePos = pSocketItem->sceneBoundingRect().center();
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
        m_inSockets[sockName].socket->toggle(bSelected);
    } else {
        ZASSERT_EXIT(m_outSockets.find(sockName) != m_outSockets.end());
        m_outSockets[sockName].socket->toggle(bSelected);
    }
}

void ZenoNode::markError(bool isError)
{
    m_bError = isError;
    if (m_bError)
        m_headerWidget->setColors(false, QColor(200, 84, 79), QColor(), QColor());
    else
        m_headerWidget->setColors(false, QColor(83, 96, 147), QColor(), QColor());
    update();
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
            ZASSERT_EXIT(m_inSockets.find(portName) != m_inSockets.end(), QPointF());
            QPointF pos = m_inSockets[portName].socket->sceneBoundingRect().center();
            return pos;
        } else {
            ZASSERT_EXIT(m_outSockets.find(portName) != m_outSockets.end(), QPointF());
            QPointF pos = m_outSockets[portName].socket->sceneBoundingRect().center();
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
    scene()->clearSelection();
    this->setSelected(true);

	QMenu* nodeMenu = new QMenu;
	QAction* pCopy = new QAction("Copy");
	QAction* pPaste = new QAction("Paste");
	QAction* pDelete = new QAction("Delete");
    QAction* pResolve = new QAction("Resolve");

    connect(pResolve, &QAction::triggered, this, [=]() { markError(false); });

	nodeMenu->addAction(pCopy);
	nodeMenu->addAction(pPaste);
	nodeMenu->addAction(pDelete);
    nodeMenu->addAction(pResolve);

    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    const QString& name = m_index.data(ROLE_OBJNAME).toString();
    QModelIndex subnetnodeIdx = pGraphsModel->index(name);
    if (subnetnodeIdx.isValid())
    {
        QAction* pFork = new QAction("Fork");
        nodeMenu->addAction(pFork);
        connect(pFork, &QAction::triggered, this, [=]()
        {
            pGraphsModel->fork(m_subGpIndex, index());
        });
    }

	nodeMenu->exec(QCursor::pos());
    nodeMenu->deleteLater();
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
	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    bool bCollasped = pGraphsModel->data2(m_subGpIndex, m_index, ROLE_COLLASPED).toBool();

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

    int options = pGraphsModel->data2(m_subGpIndex, m_index, ROLE_OPTIONS).toInt();
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

void ZenoNode::onCollaspeLegacyUpdated(bool collasped)
{
    if (collasped)
    {
        m_headerWidget->hide();
        m_bodyWidget->hide();
        //socket icon item is out of the layout.
        for (auto p : m_inSockets) {
            p.socket->hide();
        }
        for (auto p : m_outSockets) {
            p.socket->hide();
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
        for (auto p : m_inSockets) {
            p.socket->show();
        }
        for (auto p : m_outSockets) {
            p.socket->show();
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
        //socket icon item is out of the layout.
        for (auto p : m_inSockets) {
            p.socket->hide();
        }
        for (auto p : m_outSockets) {
            p.socket->hide();
        }
        m_pMainLayout->setSpacing(0);
    }
    else
    {
		m_bodyWidget->show();
        for (auto p : m_inSockets) {
            p.socket->show();
        }
        for (auto p : m_outSockets) {
            p.socket->show();
        }
        m_pMainLayout->setSpacing(1);
    }
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
        QPointF pos = value.toPointF();
        int x = pos.x(), y = pos.y();
        x = x - x % SCENE_GRID_SIZE;
        y = y - y % SCENE_GRID_SIZE;
        return QPointF(x, y);
    }
    else if (change == QGraphicsItem::ItemPositionHasChanged)
    {
        IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
        QPointF newPos = value.toPointF();
        QPointF oldPos = pGraphsModel->getNodeStatus(nodeId(), ROLE_OBJPOS, m_subGpIndex).toPointF();
        if (newPos != oldPos)
        {
            STATUS_UPDATE_INFO info;
            info.role = ROLE_OBJPOS;
            info.newValue = newPos;
            info.oldValue = oldPos;
            pGraphsModel->updateNodeStatus(nodeId(), info, m_subGpIndex, false);
        }
    }
    return value;
}
