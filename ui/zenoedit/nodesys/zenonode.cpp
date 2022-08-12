#include "zenonode.h"
#include "zenosubgraphscene.h"
#include <zenoui/model/modelrole.h>
#include "model/subgraphmodel.h"
#include <zenoui/render/common_id.h>
#include <zenoui/comctrl/gv/zenoparamnameitem.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/include/igraphsmodel.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/scope_exit.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenoui/model/variantptr.h>
#include "curvemap/zcurvemapeditor.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "graphsmanagment.h"
#include "../nodesview/zenographseditor.h"
#include "util/log.h"
#include "zenosubgraphview.h"
#include "../panel/zenoheatmapeditor.h"
#include "zvalidator.h"
#include "zenonewmenu.h"
#include "util/apphelper.h"


static QString getOpenFileName(
    const QString &caption,
    const QString &dir,
    const QString &filter
) {
    QString path = QFileDialog::getOpenFileName(nullptr, caption, dir, filter);
#if 0 // cannot work for now, wait for StringEval to be integrated into string param edit (luzh job)
    QSettings settings("ZenusTech", "Zeno");
    QVariant nas_loc_v = settings.value("nas_loc");
    path.replace('\\', '/');
    if (!nas_loc_v.isNull()) {
        QString nas = nas_loc_v.toString();
        if (!nas.isEmpty()) {
            nas.replace('\\', '/');
            path.replace(nas, "$NASLOC");
        }
    }
#endif
    return path;
}

static QString getSaveFileName(
    const QString &caption,
    const QString &dir,
    const QString &filter
) {
    QString path = QFileDialog::getSaveFileName(nullptr, caption, dir, filter);
#if 0 // cannot work for now, wait for StringEval to be integrated into string param edit (luzh job)
    QSettings settings("ZenusTech", "Zeno");
    QVariant nas_loc_v = settings.value("nas_loc");
    path.replace('\\', '/');
    if (!nas_loc_v.isNull()) {
        QString nas = nas_loc_v.toString();
        if (!nas.isEmpty()) {
            nas.replace('\\', '/');
            path.replace(nas, "$NASLOC");
        }
    }
#endif
    return path;
}

ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_previewItem(nullptr)
    , m_previewText(nullptr)
    , m_pMainLayout(nullptr)
    , m_pInSocketsLayout(nullptr)
    , m_pOutSocketsLayout(nullptr)
    , m_pSocketsLayout(nullptr)
    , m_border(new QGraphicsRectItem)
    , m_NameItem(nullptr)
    , m_bError(false)
    , m_bEnableSnap(false)
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
    }

    emit inSocketPosChanged();
    emit outSocketPosChanged();
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
    m_index = QPersistentModelIndex(index);
    m_subGpIndex = QPersistentModelIndex(subGIdx);
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    m_headerWidget = initHeaderStyle();
    m_bodyWidget = initBodyWidget();
    m_previewItem = initPreview();

    m_pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pMainLayout->addItem(m_headerWidget);
    m_pMainLayout->addItem(m_bodyWidget);
    m_pMainLayout->addItem(m_previewItem);
    m_pMainLayout->setContentsMargins(0, 0, 0, 0);
    m_pMainLayout->setSpacing(1);

    setLayout(m_pMainLayout);

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

    m_border->setZValue(ZVALUE_NODE_BORDER);
    m_border->hide();
    m_previewItem->hide();
}

ZenoBackgroundWidget* ZenoNode::initPreview()
{
    QString text = m_index.data(ROLE_OBJNAME).toString();
    QFont font = m_renderParams.nameFont;
    font.setPointSize(24);

    QColor clrBg(76, 159, 244);
    QColor clrFore(255, 255, 255);
    //clrBg = m_renderParams.headerBg.clr_normal;
    //clrFore = m_renderParams.nameClr.color();

    auto headerBg = m_renderParams.headerBg;
    ZenoBackgroundWidget *pBg = new ZenoBackgroundWidget;
    pBg->setRadius(0, 0, 0, 0);
    pBg->setColors(false, clrBg, clrBg, clrBg);
    pBg->setBorder(headerBg.border_witdh, headerBg.clr_border);

    QGraphicsLinearLayout* pVLayout = new QGraphicsLinearLayout(Qt::Vertical);

    m_previewText = new ZenoTextLayoutItem(text, font, clrFore);
    m_previewText->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

    pVLayout->addItem(m_previewText);
    pBg->setLayout(pVLayout);
    
    return pBg;
}

ZenoBackgroundWidget* ZenoNode::initHeaderStyle()
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
    int margins = ZenoStyle::dpiScaled(5);
	pNameLayout->setContentsMargins(margins, margins, margins, margins);

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
	return headerWidget;
}

ZenoBackgroundWidget* ZenoNode::initBodyWidget()
{
    ZenoBackgroundWidget *bodyWidget = new ZenoBackgroundWidget(this);

    bodyWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal, bodyBg.clr_hovered, bodyBg.clr_selected);
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);
    pVLayout->setContentsMargins(0, ZenoStyle::dpiScaled(5), 0, ZenoStyle::dpiScaled(5));

    if (QGraphicsLayout* pParamsLayout = initParams())
    {
        pParamsLayout->setContentsMargins(
            ZenoStyle::dpiScaled(m_renderParams.distParam.paramsLPadding),
            ZenoStyle::dpiScaled(10),
            ZenoStyle::dpiScaled(10),
            ZenoStyle::dpiScaled(0));
        pVLayout->addItem(pParamsLayout);
    }
    if (QGraphicsLayout* pSocketsLayout = initSockets())
    {
        pSocketsLayout->setContentsMargins(
            ZenoStyle::dpiScaled(m_renderParams.distParam.paramsLPadding),
            ZenoStyle::dpiScaled(m_renderParams.distParam.paramsToTopSocket),
            ZenoStyle::dpiScaled(m_renderParams.distParam.paramsLPadding),
            ZenoStyle::dpiScaled(m_renderParams.distParam.outSocketsBottomMargin));
        pVLayout->addItem(pSocketsLayout);
    }

    bodyWidget->setLayout(pVLayout);
    return bodyWidget;
}

QGraphicsLayout* ZenoNode::initParams()
{
    const PARAMS_INFO &params = m_index.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
    QList<QString> names = params.keys();
    int n = names.length();
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

            QGraphicsLayout* pParamLayout = initParam(param.control, paramName, param);
            if (pParamLayout)
            {
                pParamsLayout->addItem(pParamLayout);
            }
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

QGraphicsLayout* ZenoNode::initParam(PARAM_CONTROL ctrl, const QString& paramName, const PARAM_INFO& param)
{
    if (ctrl == CONTROL_NONVISIBLE)
        return nullptr;

    QGraphicsLinearLayout* pParamLayout = new QGraphicsLinearLayout(Qt::Horizontal);
    const QString& value = UiHelper::variantToString(param.value);

	ZenoTextLayoutItem* pNameItem = new ZenoTextLayoutItem(paramName, m_renderParams.paramFont, m_renderParams.paramClr.color());
	pParamLayout->addItem(pNameItem);

	switch (param.control)
	{
	    case CONTROL_STRING:
	    case CONTROL_INT:
	    case CONTROL_FLOAT:
	    {
		    ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(value, param.control,  m_renderParams.lineEditParam);
            pLineEdit->setValidator(validateForParams(param));
		    pParamLayout->addItem(pLineEdit);
		    connect(pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(paramName, pLineEdit->text());
			    });
		    m_paramControls[paramName] = pLineEdit;
		    break;
	    }
        case CONTROL_BOOL:
        {
            ZenoParamCheckBox *pSocketCheckbox = new ZenoParamCheckBox(paramName);
            pParamLayout->addItem(pSocketCheckbox);

            bool isChecked = param.value.toBool();
            pSocketCheckbox->setCheckState(isChecked ? Qt::Checked : Qt::Unchecked);

            connect(pSocketCheckbox, &ZenoParamCheckBox::stateChanged, this, [paramName, this](int state) {
                bool bChecked = false;
                if (state == Qt::Checked) {
                    bChecked = true;
                }
                else if (state == Qt::Unchecked) {
                    bChecked = false;
                }
                else {
                    Q_ASSERT(false);
                    return;
                }

                IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pGraphsModel);

                const QString& nodeid = m_index.data(ROLE_OBJID).toString();

                PARAM_UPDATE_INFO info;
                info.oldValue = pGraphsModel->getParamValue(nodeid, paramName, m_subGpIndex);
                info.newValue = bChecked;
                info.name = paramName;
                if (info.oldValue != info.newValue)
                    pGraphsModel->updateParamInfo(nodeid, info, m_subGpIndex, true);

                updateWhole();
            });
            m_paramControls[paramName] = pSocketCheckbox;
            break;
        }
        case CONTROL_VEC3:
        {
            UI_VECTYPE vec = param.value.value<UI_VECTYPE>();
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
            pComboBox->setText(value);
		    pParamLayout->addItem(pComboBox);

		    connect(pComboBox, &ZenoParamComboBox::textActivated, this, [paramName, this](const QString& textValue) {
                onParamEditFinished(paramName, textValue);
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
			    onParamEditFinished(paramName, pFileWidget->text());
			});
            connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                DlgInEventLoopScope;
                QString path = getOpenFileName("File to Open", "", "All Files(*);;");
                if (path.isEmpty())
                    return;
                pFileWidget->setText(path);
                onParamEditFinished(paramName, path);
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
			    onParamEditFinished(paramName, pFileWidget->text());
			});
            connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                DlgInEventLoopScope;
                QString path = getSaveFileName(tr("Path to Save"), "", "All Files(*);;");
                if (path.isEmpty())
                    return;
                pFileWidget->setText(path);
                onParamEditFinished(paramName, path);
            });
            m_paramControls[paramName] = pFileWidget;
		    break;
	    }
	    case CONTROL_MULTILINE_STRING:
	    {
		    ZenoParamMultilineStr* pMultiStrEdit = new ZenoParamMultilineStr(value, m_renderParams.lineEditParam);
		    pParamLayout->addItem(pMultiStrEdit);
		    connect(pMultiStrEdit, &ZenoParamMultilineStr::editingFinished, this, [=]() {
			    onParamEditFinished(paramName, pMultiStrEdit->text());
			    });
		    m_paramControls[paramName] = pMultiStrEdit;
		    break;
	    }
	    case CONTROL_COLOR:
        {
            break;
        }
	    case CONTROL_CURVE:
	    {
            ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
		    pParamLayout->addItem(pEditBtn);
            connect(pEditBtn, &ZenoParamPushButton::clicked, this, [=]() {
                IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pGraphsModel);
                CurveModel *pModel = nullptr;
                const QVariant& val = pGraphsModel->getParamValue(m_index.data(ROLE_OBJID).toString(), paramName, m_subGpIndex);
                pModel = QVariantPtr<CurveModel>::asPtr(val);
                if (!pModel)
                {
                    pModel = curve_util::deflModel(pGraphsModel);
                    onParamEditFinished(paramName, QVariantPtr<CurveModel>::asVariant(pModel));
                }
                ZASSERT_EXIT(pModel);
                ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);
                pEditor->setAttribute(Qt::WA_DeleteOnClose);
                pEditor->addCurve(pModel);
                pEditor->show();
            });
		    m_paramControls[paramName] = pEditBtn;
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
    return pParamLayout;
}

QPersistentModelIndex ZenoNode::subGraphIndex() const
{
    return m_subGpIndex;
}

QValidator* ZenoNode::validateForParams(PARAM_INFO info)
{
    switch (info.control)
    {
    case CONTROL_INT:       return new QIntValidator;
    case CONTROL_FLOAT:     return new QDoubleValidator;
    case CONTROL_READPATH:
    case CONTROL_WRITEPATH:
        return new PathValidator;
    }
    return nullptr;
}

QValidator* ZenoNode::validateForSockets(INPUT_SOCKET inSocket)
{
    switch (inSocket.info.control)
    {
    case CONTROL_INT:       return new QIntValidator;
    case CONTROL_FLOAT:     return new QDoubleValidator;
    case CONTROL_READPATH:
    case CONTROL_WRITEPATH: return new PathValidator;
    }
    return nullptr;
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
        info.newValue = value;
        break;
    default:
        info.newValue = UiHelper::parseTextValue(param.control, value.toString());
        break;
    }
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
            pVecEdit->setVec(val.value<UI_VECTYPE>());
        }
        else if (ZenoParamCheckBox* pCheckbox = qobject_cast<ZenoParamCheckBox*>(pWidget))
        {
            pCheckbox->setCheckState(val.toBool() ? Qt::Checked : Qt::Unchecked);
            updateWhole();
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

void ZenoNode::clearInSocketControl(const QString& sockName)
{
    ZASSERT_EXIT(m_inSockets.find(sockName) != m_inSockets.end());
    _socket_ctrl ctrl = m_inSockets[sockName];
    QGraphicsLinearLayout* pControlLayout = ctrl.ctrl_layout;
    if (ctrl.socket_control) {
        pControlLayout->removeItem(ctrl.socket_control);
        delete ctrl.socket_control;
        m_inSockets[sockName].socket_control = nullptr;
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
            ZASSERT_EXIT(oldName != newName, false);

            _socket_ctrl ctrl = m_inSockets[oldName];
            ctrl.socket_text->setText(newName);
            ctrl.socket->updateSockName(newName);

            //have to reset the text format, which is trivial.
            QTextFrame *frame = ctrl.socket_text->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            m_inSockets[newName] = ctrl;
            m_inSockets.remove(oldName);

            updateWhole();
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
            ZASSERT_EXIT(oldName != newName, false);

            _socket_ctrl ctrl = m_outSockets[oldName];
            ctrl.socket_text->setText(newName);
            ctrl.socket->updateSockName(newName);

            //have to reset the text format, which is trivial.
            QTextFrame *frame = ctrl.socket_text->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            m_outSockets[newName] = ctrl;
            m_outSockets.remove(oldName);

            updateWhole();
            return true;
        }
        return false;
    }
}

void ZenoNode::onSocketsUpdate(bool bInput, bool bInit)
{
    const QString &nodeid = nodeId();
    const QString& nodeName = m_index.data(ROLE_OBJNAME).toString();

    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pModel);

    if (bInput)
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();

        if (renameDictKey(true, inputs, OUTPUT_SOCKETS()))
        {
            return;
        }
        for (const INPUT_SOCKET inSocket : inputs)
        {
            const QString& inSock = inSocket.info.name;
            if (m_inSockets.find(inSock) == m_inSockets.end())
            {
                //add socket
                ZenoSocketItem *socket = new ZenoSocketItem(inSock, true, m_index, m_renderParams.socket, m_renderParams.szSocket, this);
                socket->updateSockName(inSock);
                socket->setZValue(ZVALUE_ELEMENT);
                connect(socket, &ZenoSocketItem::clicked, this, [=]() {
                    QString sockName;
                    QPointF sockPos;
                    bool bInput = false;
                    QPersistentModelIndex linkIdx;
                    getSocketInfoByItem(socket, sockName, sockPos, bInput, linkIdx);
                    // must use sockName rather than outSock because the sockname is dynamic.
                    emit socketClicked(nodeid, bInput, sockName, sockPos, linkIdx);
                });

                QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);

                ZenoTextLayoutItem* pSocketItem = new ZenoTextLayoutItem(inSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
                pMiniLayout->addItem(pSocketItem);

                const QString &sockType = inSocket.info.type;
                PARAM_CONTROL ctrl = inSocket.info.control;

                ZenoParamWidget* pSocketControl = initSocketWidget(inSocket, pSocketItem);
                if (pSocketControl) {
                    pMiniLayout->addItem(pSocketControl);
                    pSocketControl->setVisible(inSocket.linkIndice.isEmpty());
                }

                _socket_ctrl socket_ctrl;
                socket_ctrl.socket = socket;
                socket_ctrl.socket_text = pSocketItem;
                socket_ctrl.socket_control = pSocketControl;
                socket_ctrl.ctrl_layout = pMiniLayout;

                m_inSockets.insert(inSock, socket_ctrl);

                if (!bInit && pModel->IsSubGraphNode(m_index))
                {
                    //dynamic socket added, ensure that the key is above the SRC key.
                    m_pInSocketsLayout->insertItem(0, pMiniLayout);
                }
                else
                {
                    m_pInSocketsLayout->addItem(pMiniLayout);
                }
                updateWhole();
            }
            else
            {
                //update value directly.
                updateSocketWidget(inSocket);
            }
        }
        //remove all keys which don't exist in model.
        for (auto name : m_inSockets.keys())
        {
            if (inputs.find(name) == inputs.end())
            {
                const _socket_ctrl &sock = m_inSockets[name];
                //auto socketlayout = sock.socket_text->parentLayoutItem();
                m_pInSocketsLayout->removeItem(sock.ctrl_layout);
                delete sock.socket;
                delete sock.socket_text;
                m_inSockets.remove(name);
                updateWhole();
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
                _socket_ctrl sock;
                sock.socket = new ZenoSocketItem(outSock, false, m_index, m_renderParams.socket, m_renderParams.szSocket, this);
                sock.socket->updateSockName(outSock);
                sock.socket_text = new ZenoTextLayoutItem(outSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
                sock.socket_text->setRight(true);
                connect(sock.socket, &ZenoSocketItem::clicked, this, [=]() {
                    QString sockName;
                    QPointF sockPos;
                    bool bInput = false;
                    QPersistentModelIndex linkIdx;
                    getSocketInfoByItem(sock.socket, sockName, sockPos, bInput, linkIdx);
                    // must use sockName rather than outSock because the sockname is dynamic.
                    emit socketClicked(nodeid, bInput, sockName, sockPos, linkIdx);
                });

                if (outSocket.info.control == CONTROL_DICTKEY)
                {
                    sock.socket_text->setTextInteractionFlags(Qt::TextEditorInteraction);

                    QTextFrame *frame = sock.socket_text->document()->rootFrame();
                    QTextFrameFormat format = frame->frameFormat();
                    format.setBackground(QColor(37, 37, 37));
                    frame->setFrameFormat(format);

                    connect(sock.socket_text, &ZenoTextLayoutItem::contentsChanged, this, [=](QString oldText, QString newText) {
                        IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                        ZASSERT_EXIT(pGraphsModel);
                        SOCKET_UPDATE_INFO info;
                        info.bInput = false;
                        info.newInfo.name = newText;
                        info.oldInfo.name = oldText;
                        info.updateWay = SOCKET_UPDATE_NAME;
                        bool ret = pGraphsModel->updateSocketNameNotDesc(m_index.data(ROLE_OBJID).toString(), info, m_subGpIndex, true);
                        if (!ret) {
                            //todo: error hint.
                            sock.socket_text->blockSignals(true);
                            sock.socket_text->setText(oldText);
                            sock.socket_text->blockSignals(false);
                        }
                    });
                    sock.socket_control = nullptr;
                }

                m_outSockets[outSock] = sock;

                QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);
                pMiniLayout->addItem(sock.socket_text);

				if (!bInit && pModel->IsSubGraphNode(m_index))
				{
					//dynamic socket added, ensure that the key is above the DST key.
                    m_pOutSocketsLayout->insertItem(0, pMiniLayout);
				}
				else
				{
                    m_pOutSocketsLayout->addItem(pMiniLayout);
				}

                updateWhole();
            }
        }

        //remove all keys which don't exist in model.
        for (auto name : m_outSockets.keys())
        {
            if (outputs.find(name) == outputs.end())
            {
                const _socket_ctrl &sock = m_outSockets[name];
                //ugly...
                auto socketlayout = sock.socket_text->parentLayoutItem();
                m_pOutSocketsLayout->removeItem(socketlayout);
                delete sock.socket;
                delete sock.socket_text;
                m_outSockets.remove(name);
                updateWhole();
            }
        }
    }
}

ZenoParamWidget* ZenoNode::initSocketWidget(const INPUT_SOCKET inSocket, ZenoTextLayoutItem* pSocketText)
{
    const QString& nodeid = nodeId();
    const QString& inSock = inSocket.info.name;
    const QString& sockType = inSocket.info.type;
    PARAM_CONTROL ctrl = inSocket.info.control;
    switch (ctrl)
    {
        case CONTROL_INT:
        case CONTROL_FLOAT:
        case CONTROL_STRING:
        {
            ZenoParamLineEdit *pSocketEditor = new ZenoParamLineEdit(
                UiHelper::variantToString(inSocket.info.defaultValue),
                inSocket.info.control,
                m_renderParams.lineEditParam);
            pSocketEditor->setValidator(validateForSockets(inSocket));
            //todo: allow to edit path directly?
            connect(pSocketEditor, &ZenoParamLineEdit::editingFinished, this, [=]() {

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                const QVariant &newValue = UiHelper::_parseDefaultValue(pSocketEditor->text(), _inSocket.info.type);
                updateSocketDeflValue(nodeid, inSock, _inSocket, newValue);
            });
            return pSocketEditor;
        }
        case CONTROL_BOOL:
        {
            ZenoParamCheckBox* pSocketCheckbox = new ZenoParamCheckBox(inSock);
            bool isChecked = inSocket.info.defaultValue.toBool();
            pSocketCheckbox->setCheckState(isChecked ? Qt::Checked : Qt::Unchecked);
            connect(pSocketCheckbox, &ZenoParamCheckBox::stateChanged, this, [=](int state) {
                bool bChecked = false;
                if (state == Qt::Checked) {
                    bChecked = true;
                }
                else if (state == Qt::Unchecked) {
                    bChecked = false;
                }
                else {
                    Q_ASSERT(false);
                    return;
                }

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                updateSocketDeflValue(nodeid, inSock, _inSocket, bChecked);
            });
            return pSocketCheckbox;
        }
        case CONTROL_READPATH:
        case CONTROL_WRITEPATH:
        {
            const QString& path = UiHelper::variantToString(inSocket.info.defaultValue);
            ZenoParamPathEdit *pPathEditor = new ZenoParamPathEdit(path, ctrl, m_renderParams.lineEditParam);
            bool isRead = ctrl == CONTROL_READPATH;

            connect(pPathEditor, &ZenoParamPathEdit::clicked, this, [=]() {
                DlgInEventLoopScope;
                QString path;
                if (isRead) {
                    path = getOpenFileName(tr("File to Open"), "", tr("All Files(*);;"));
                } else {
                    path = getSaveFileName(tr("Path to Save"), "", tr("All Files(*);;"));
                }
                if (path.isEmpty())
                    return;
                pPathEditor->setPath(path);
            });
            connect(pPathEditor, &ZenoParamPathEdit::pathValueChanged, this, [=](QString newPath) {

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                updateSocketDeflValue(nodeid, inSock, _inSocket, newPath);
            });
            return pPathEditor;
        }
        case CONTROL_MULTILINE_STRING:
        {
            //todo
            break;
        }
        case CONTROL_COLOR:
        {
            QLinearGradient grad = inSocket.info.defaultValue.value<QLinearGradient>();
            ZenoParamPushButton* pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
            connect(pEditBtn, &ZenoParamPushButton::clicked, this, [=]() {
                ZenoHeatMapEditor editor(grad);
                editor.exec();
                QLinearGradient newGrad = editor.colorRamps();

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                updateSocketDeflValue(nodeid, inSock, _inSocket, QVariant::fromValue(newGrad));
            });
            return pEditBtn;
        }
        case CONTROL_DICTKEY:
        {
            pSocketText->setTextInteractionFlags(Qt::TextEditorInteraction);

            QTextFrame *frame = pSocketText->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            connect(pSocketText, &ZenoTextLayoutItem::contentsChanged, this, [=](QString oldText, QString newText) {
                IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                ZASSERT_EXIT(pGraphsModel);
                SOCKET_UPDATE_INFO info;
                info.bInput = true;
                info.newInfo.name = newText;
                info.oldInfo.name = oldText;
                info.updateWay = SOCKET_UPDATE_NAME;
                bool ret = pGraphsModel->updateSocketNameNotDesc(m_index.data(ROLE_OBJID).toString(), info,
                                                                 m_subGpIndex, true);
                if (!ret) {
                    //todo: error hint.
                    pSocketText->blockSignals(true);
                    pSocketText->setText(oldText);
                    pSocketText->blockSignals(false);
                }
            });
            return nullptr; //no control expect key editor.
        }
        case CONTROL_VEC3:
        {
            UI_VECTYPE vec = inSocket.info.defaultValue.value<UI_VECTYPE>();
            ZenoVecEditWidget *pVecEditor = new ZenoVecEditWidget(vec);
            connect(pVecEditor, &ZenoVecEditWidget::editingFinished, this, [=]() {
                UI_VECTYPE vec = pVecEditor->vec();
                const QVariant &newValue = QVariant::fromValue(vec);

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                updateSocketDeflValue(nodeid, inSock, _inSocket, newValue);
            });
            return pVecEditor;
        }
        case CONTROL_ENUM:
        {
            QStringList items = sockType.mid(QString("enum ").length()).split(QRegExp("\\s+"));
            ZenoParamComboBox *pComboBox = new ZenoParamComboBox(items, m_renderParams.comboboxParam);
            QString val = inSocket.info.defaultValue.toString();
            if (items.indexOf(val) != -1)
            {
                pComboBox->setText(val);
            }
            connect(pComboBox, &ZenoParamComboBox::textActivated, this, [=](const QString &textValue) {
                QString oldValue = pComboBox->text();

                bool bOk = false;
                INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                ZASSERT_EXIT(bOk);

                updateSocketDeflValue(nodeid, inSock, _inSocket, textValue);
            });
            return pComboBox;
        }
        case CONTROL_CURVE:
        {
            ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
            connect(pEditBtn, &ZenoParamPushButton::clicked, this, [=]() {
                INPUT_SOCKETS _inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
                const QVariant &oldValue = _inputs[inSock].info.defaultValue;
                ZCurveMapEditor *pEditor = new ZCurveMapEditor(true);
                pEditor->setAttribute(Qt::WA_DeleteOnClose);
                CurveModel *pModel = QVariantPtr<CurveModel>::asPtr(oldValue);
                if (!pModel) {
                    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                    ZASSERT_EXIT(pGraphsModel);
                    pModel = curve_util::deflModel(pGraphsModel);
                }
                ZASSERT_EXIT(pModel);
                pEditor->addCurve(pModel);
                pEditor->show();

                connect(pEditor, &ZCurveMapEditor::finished, this, [=](int result) {
                    ZASSERT_EXIT(pEditor->curveCount() == 1);
                    CurveModel *pCurveModel = pEditor->getCurve(0);
                    const QVariant &newValue = QVariantPtr<CurveModel>::asVariant(pCurveModel);

                    bool bOk = false;
                    INPUT_SOCKET _inSocket = AppHelper::getInputSocket(m_index, inSock, bOk);
                    ZASSERT_EXIT(bOk);

                    updateSocketDeflValue(nodeid, inSock, _inSocket, newValue);
                });
            });
            return pEditBtn;
        }
        default:
            return nullptr;
    }
    return nullptr;
}

void ZenoNode::updateSocketWidget(const INPUT_SOCKET inSocket)
{
    _socket_ctrl ctrl = m_inSockets[inSocket.info.name];
    QGraphicsLinearLayout *pControlLayout = ctrl.ctrl_layout;
    ZASSERT_EXIT(pControlLayout);

    bool bUpdateLayout = false;
    switch (inSocket.info.control)
    {
        case CONTROL_STRING:
        case CONTROL_INT:
        case CONTROL_FLOAT:
        {
            ZenoParamLineEdit *pLineEdit = qobject_cast<ZenoParamLineEdit*>(ctrl.socket_control);
            if (!pLineEdit) {
                //sock type has been changed to this control type
                clearInSocketControl(inSocket.info.name);
                pLineEdit = qobject_cast<ZenoParamLineEdit *>(initSocketWidget(inSocket, ctrl.socket_text));
                ZASSERT_EXIT(pLineEdit);
                pControlLayout->addItem(pLineEdit);
                m_inSockets[inSocket.info.name].socket_control = pLineEdit;
                bUpdateLayout = true;
            }
            pLineEdit->setValidator(validateForSockets(inSocket));
            pLineEdit->setText(inSocket.info.defaultValue.toString());
            break;
        }
        case CONTROL_BOOL:
        {
            ZenoParamCheckBox *pSocketCheckbox = qobject_cast<ZenoParamCheckBox *>(ctrl.socket_control);
            if (!pSocketCheckbox) {
                clearInSocketControl(inSocket.info.name);
                pSocketCheckbox = qobject_cast<ZenoParamCheckBox *>(initSocketWidget(inSocket, ctrl.socket_text));
                ZASSERT_EXIT(pSocketCheckbox);
                pControlLayout->addItem(pSocketCheckbox);
                m_inSockets[inSocket.info.name].socket_control = pSocketCheckbox;
                bUpdateLayout = true;
            }
            bool bChecked = inSocket.info.defaultValue.toBool();
            pSocketCheckbox->setCheckState(bChecked ? Qt::Checked : Qt::Unchecked);
            break;
        }
        case CONTROL_VEC3:
        {
            ZenoVecEditWidget *pVecEdit = qobject_cast<ZenoVecEditWidget *>(ctrl.socket_control);
            if (!pVecEdit) {
                clearInSocketControl(inSocket.info.name);
                pVecEdit = qobject_cast<ZenoVecEditWidget *>(initSocketWidget(inSocket, ctrl.socket_text));
                ZASSERT_EXIT(pVecEdit);
                pControlLayout->addItem(pVecEdit);
                m_inSockets[inSocket.info.name].socket_control = pVecEdit;
                bUpdateLayout = true;
            }
            const UI_VECTYPE& vec = inSocket.info.defaultValue.value<UI_VECTYPE>();
            pVecEdit->setVec(vec);
            break;
        }
        case CONTROL_ENUM:
        {
            ZenoParamComboBox *pComboBox = qobject_cast<ZenoParamComboBox *>(ctrl.socket_control);
            if (!pComboBox) {
                clearInSocketControl(inSocket.info.name);
                pComboBox = qobject_cast<ZenoParamComboBox *>(initSocketWidget(inSocket, ctrl.socket_text));
                ZASSERT_EXIT(pComboBox);
                pControlLayout->addItem(pComboBox);
                m_inSockets[inSocket.info.name].socket_control = pComboBox;
                bUpdateLayout = true;
            }
            pComboBox->setText(inSocket.info.defaultValue.toString());
            break;
        }
        case CONTROL_NONE: {
            //should clear the control if exists.
            if (ctrl.socket_control) {
                clearInSocketControl(inSocket.info.name);
                bUpdateLayout = true;
            }
            break;
        }
    }
    if (bUpdateLayout) {
        updateWhole();
    }
}

void ZenoNode::updateWhole()
{
    if (m_pInSocketsLayout)
        m_pInSocketsLayout->invalidate();
    if (m_pOutSocketsLayout)
        m_pOutSocketsLayout->invalidate();
    if (m_pSocketsLayout)
        m_pSocketsLayout->invalidate();
    if (m_bodyWidget)
        m_bodyWidget->layout()->invalidate();
    if (m_headerWidget)
        m_headerWidget->layout()->invalidate();
    if (m_previewItem)
        m_previewItem->layout()->invalidate();
    if (m_pMainLayout)
        m_pMainLayout->invalidate();
    this->updateGeometry();
}

void ZenoNode::onSocketLinkChanged(const QString& sockName, bool bInput, bool bAdded)
{
	if (bInput)
	{
        if (m_inSockets.find(sockName) != m_inSockets.end())
        {
            m_inSockets[sockName].socket->toggle(bAdded);
            m_inSockets[sockName].socket->setSockStatus(bAdded ? ZenoSocketItem::STATUS_CONNECTED : ZenoSocketItem::STATUS_NOCONN);
            if (m_inSockets[sockName].socket_control)
                m_inSockets[sockName].socket_control->setVisible(!bAdded);
        }
	}
	else
	{
        if (m_outSockets.find(sockName) != m_outSockets.end())
        {
            m_outSockets[sockName].socket->toggle(bAdded);
            m_outSockets[sockName].socket->setSockStatus(bAdded ? ZenoSocketItem::STATUS_CONNECTED : ZenoSocketItem::STATUS_NOCONN);
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

QGraphicsLayout* ZenoNode::initSockets()
{
    m_pSocketsLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pInSocketsLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pOutSocketsLayout = new QGraphicsLinearLayout(Qt::Vertical);
    onSocketsUpdate(true, true);
    onSocketsUpdate(false, true);
    m_pSocketsLayout->addItem(m_pInSocketsLayout);
    m_pSocketsLayout->addItem(m_pOutSocketsLayout);

    qreal right, top, left, bottom;
    m_pSocketsLayout->getContentsMargins(&left, &top, &right, &bottom);
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

void ZenoNode::switchView(bool bPreview)
{
    if (!bEnableZoomPreview)
        return;
    
    m_headerWidget->setVisible(!bPreview);
    m_bodyWidget->setVisible(!bPreview);
    m_previewItem->setVisible(bPreview);
    m_pStatusWidgets->setVisible(!bPreview);

    for (auto p : m_inSockets) {
        p.socket->setVisible(!bPreview);
    }
    for (auto p : m_outSockets) {
        p.socket->setVisible(!bPreview);
    }
    adjustPreview(bPreview);
}

void ZenoNode::adjustPreview(bool bVisible)
{
    ZASSERT_EXIT(m_previewText);
    QFont font = m_previewText->font();
    if (bVisible) {
        font.setPointSize(100);
    }
    else {
        font.setPointSize(24);
    }
    m_previewText->setFont(font);
    m_previewText->updateGeometry();

    qreal leftM = bVisible ? ZenoStyle::dpiScaled(100) : 0;
    qreal rightM = leftM;
    qreal topM = bVisible ? ZenoStyle::dpiScaled(50) : 0;
    qreal bottomM = topM;
    m_previewText->setMargins(leftM, topM, rightM, bottomM);

    updateWhole();
}

void ZenoNode::setGeometry(const QRectF &rect)
{
    QRectF rc = rect;
    if (bEnableZoomPreview) {
        // solution for preview node case
        QSizeF sz = effectiveSizeHint(Qt::MinimumSize);
        rc.setSize(sz);
    }
    _base::setGeometry(rc);
}

QSizeF ZenoNode::sizeHint(Qt::SizeHint which, const QSizeF &constraint) const
{
    QSizeF sz = _base::sizeHint(which, constraint);
    return sz;
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
        return m_inSockets[sockName].socket;
    } else {
        ZASSERT_EXIT(m_outSockets.find(sockName) != m_outSockets.end(), nullptr);
        return m_outSockets[sockName].socket;
    }
}

ZenoSocketItem* ZenoNode::getNearestSocket(const QPointF& pos, bool bInput)
{
    ZenoSocketItem* pItem = nullptr;
    float minDist = std::numeric_limits<float>::max();
    auto socks = bInput ? m_inSockets : m_outSockets;
    for (_socket_ctrl ctrl : socks)
    {
        QPointF sockPos = ctrl.socket->sceneBoundingRect().center();
        QPointF offset = sockPos - pos;
        float dist = std::sqrt(offset.x() * offset.x() + offset.y() * offset.y());
        if (dist < minDist)
        {
            minDist = dist;
            pItem = ctrl.socket;
        }
    }
    return pItem;
}

QPointF ZenoNode::getPortPos(bool bInput, const QString &portName)
{
    //preview.
    if (m_previewItem->isVisible())
    {
        QRectF rc = m_previewItem->sceneBoundingRect();
        if (bInput) {
            return QPointF(rc.left(), rc.center().y());
        } else {
            return QPointF(rc.right(), rc.center().y());
        }
    }
    else
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
                QPointF pos = m_inSockets[portName].socket->sceneBoundingRect().center();
                return pos;
            } else {
                ZASSERT_EXIT(m_outSockets.find(portName) != m_outSockets.end(), QPointF());
                QPointF pos = m_outSockets[portName].socket->sceneBoundingRect().center();
                return pos;
            }
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
        QPointF oldPos = pGraphsModel->getNodeStatus(nodeId(), ROLE_OBJPOS, m_subGpIndex).toPointF();
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
