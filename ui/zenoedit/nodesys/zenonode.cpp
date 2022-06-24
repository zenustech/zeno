#include "zenonode.h"
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

static QString getOpenFileName(
    const QString &caption,
    const QString &dir,
    const QString &filter
) {
    QString path = QFileDialog::getOpenFileName(nullptr, caption, dir, filter);
    QSettings settings("ZenusTech", "Zeno");
    QVariant nas_loc_v = settings.value("nas_loc");
    path.replace('\\', '/');
    if (!nas_loc_v.isNull()) {
        QString nas = nas_loc_v.toString();
        nas.replace('\\', '/');
        path.replace(nas, "$NASLOC");
    }
    return path;
}

static QString getSaveFileName(
    const QString &caption,
    const QString &dir,
    const QString &filter
) {
    QString path = QFileDialog::getSaveFileName(nullptr, caption, dir, filter);
    QSettings settings("ZenusTech", "Zeno");
    QVariant nas_loc_v = settings.value("nas_loc");
    path.replace('\\', '/');
    if (!nas_loc_v.isNull()) {
        QString nas = nas_loc_v.toString();
        nas.replace('\\', '/');
        path.replace(nas, "$NASLOC");
    }
    return path;
}

ZenoNode::ZenoNode(const NodeUtilParam &params, QGraphicsItem *parent)
    : _base(parent)
    , m_renderParams(params)
    , m_bInitSockets(false)
    , m_bodyWidget(nullptr)
    , m_headerWidget(nullptr)
    , m_collaspedWidget(nullptr)
    , m_bHeapMap(false)
    , m_pMainLayout(nullptr)
    , m_pInSocketsLayout(nullptr)
    , m_pOutSocketsLayout(nullptr)
    , m_pSocketsLayout(nullptr)
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
    m_index = QPersistentModelIndex(index);
    m_subGpIndex = QPersistentModelIndex(subGIdx);
    NODE_TYPE type = static_cast<NODE_TYPE>(m_index.data(ROLE_NODETYPE).toInt());

    IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);

    m_headerWidget = initHeaderWangStyle(type);
    m_bodyWidget = initBodyWidget(type);

    m_pMainLayout = new QGraphicsLinearLayout(Qt::Vertical);
    m_pMainLayout->addItem(m_headerWidget);
    m_pMainLayout->addItem(m_bodyWidget);
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

    bodyWidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);

    const auto &bodyBg = m_renderParams.bodyBg;
    bodyWidget->setRadius(bodyBg.lt_radius, bodyBg.rt_radius, bodyBg.lb_radius, bodyBg.rb_radius);
    bodyWidget->setColors(bodyBg.bAcceptHovers, bodyBg.clr_normal, bodyBg.clr_hovered, bodyBg.clr_selected);
    bodyWidget->setBorder(bodyBg.border_witdh, bodyBg.clr_border);

    QGraphicsLinearLayout *pVLayout = new QGraphicsLinearLayout(Qt::Vertical);
    pVLayout->setContentsMargins(0, 5, 0, 5);

    if (QGraphicsLayout* pParamsLayout = initParams())
    {
        pParamsLayout->setContentsMargins(m_renderParams.distParam.paramsLPadding, 10, 10, 0);
        pVLayout->addItem(pParamsLayout);
    }
    if (QGraphicsLayout* pSocketsLayout = initSockets())
    {
        pSocketsLayout->setContentsMargins(
            m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.paramsToTopSocket,
            m_renderParams.distParam.paramsLPadding, m_renderParams.distParam.outSocketsBottomMargin);
        pVLayout->addItem(pSocketsLayout);
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
	    {
		    ZenoParamLineEdit* pLineEdit = new ZenoParamLineEdit(value, param.control,  m_renderParams.lineEditParam);
		    pParamLayout->addItem(pLineEdit);
		    connect(pLineEdit, &ZenoParamLineEdit::editingFinished, this, [=]() {
			    onParamEditFinished(param.control, paramName, pLineEdit->text());
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
            pComboBox->setText(value);
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
                DlgInEventLoopScope;
                QString path = getOpenFileName("File to Open", "", "All Files(*);;");
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
                DlgInEventLoopScope;
                QString path = getSaveFileName("Path to Save", "", "All Files(*);;");
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
                    onParamEditFinished(param.control, paramName, QVariantPtr<CurveModel>::asVariant(pModel));
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
}

QPersistentModelIndex ZenoNode::subGraphIndex() const
{
    return m_subGpIndex;
}

void ZenoNode::onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QVariant& value)
{
    // graphics item sync to model.
    const QString nodeid = nodeId();
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();

    PARAM_UPDATE_INFO info;
    info.oldValue = pGraphsModel->getParamValue(nodeid, paramName, m_subGpIndex);
    info.name = paramName;

    switch (editCtrl)
    {
    case CONTROL_HEATMAP:
    case CONTROL_CURVE:
        info.newValue = value;
        break;
    default:
        info.newValue = UiHelper::parseTextValue(editCtrl, value.toString());
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
            pVecEdit->setVec(val.value<QVector<qreal>>());
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

void ZenoNode::onSocketsUpdate(bool bInput)
{
    const QString &nodeid = nodeId();
    if (bInput)
    {
        INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();

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
            ZASSERT_EXIT(oldName != newName);

            _socket_ctrl ctrl = m_inSockets[oldName];
            ctrl.socket_text->setText(newName);
            
            //have to reset the text format, which is trivial.
            QTextFrame *frame = ctrl.socket_text->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            m_inSockets[newName] = ctrl;
            m_inSockets.remove(oldName);

            updateWhole();
            return;
        }

        for (INPUT_SOCKET inSocket : inputs)
        {
            //copy from initSockets()
            const QString &inSock = inSocket.info.name;
            if (m_inSockets.find(inSock) == m_inSockets.end())
            {
                //add socket

                ZenoSocketItem *socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
                socket->setIsInput(true);
                socket->setZValue(ZVALUE_ELEMENT);

                QGraphicsLinearLayout *pMiniLayout = new QGraphicsLinearLayout(Qt::Horizontal);

                ZenoTextLayoutItem *pSocketItem =
                    new ZenoTextLayoutItem(inSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
                pMiniLayout->addItem(pSocketItem);

                _socket_ctrl socket_ctrl;

                const INPUT_SOCKET &inSocket = inputs[inSock];
                const QString &sockType = inSocket.info.type;
                PARAM_CONTROL ctrl = inSocket.info.control;

                if (ctrl == CONTROL_INT ||
                    ctrl == CONTROL_FLOAT ||
                    ctrl == CONTROL_STRING ||
                    ctrl == CONTROL_READPATH ||
                    ctrl == CONTROL_WRITEPATH)
                {
                    ZenoParamLineEdit *pSocketEditor = new ZenoParamLineEdit(UiHelper::variantToString(inSocket.info.defaultValue),
                                              inSocket.info.control, m_renderParams.lineEditParam);
                    pMiniLayout->addItem(pSocketEditor);
                    //todo: allow to edit path directly?
                    connect(pSocketEditor, &ZenoParamLineEdit::editingFinished, this, [=]() {
                        const QVariant &newValue =
                            UiHelper::_parseDefaultValue(pSocketEditor->text(), inSocket.info.type);
                        updateSocketDeflValue(nodeid, inSock, inSocket, newValue);
                    });
                    socket_ctrl.socket_control = pSocketEditor;

                    if (ctrl == CONTROL_READPATH || ctrl == CONTROL_WRITEPATH) {
                        ImageElement elem;
                        elem.image = ":/icons/ic_openfile.svg";
                        elem.imageHovered = ":/icons/ic_openfile-on.svg";
                        elem.imageOn = ":/icons/ic_openfile-on.svg";

                        bool isRead = (ctrl == CONTROL_READPATH);

                        ZenoSvgLayoutItem *openBtn = new ZenoSvgLayoutItem(elem, QSizeF(30, 30));
                        connect(openBtn, &ZenoImageItem::clicked, this, [=]() {
                            DlgInEventLoopScope;
                            QString path;
                            if (isRead) {
                                path = getOpenFileName("File to Open", "", "All Files(*);;");
                            } else {
                                path = getSaveFileName("Path to Save", "", "All Files(*);;");
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
                else if (ctrl == CONTROL_BOOL)
                {
                    ZenoParamCheckBox *pSocketCheckbox = new ZenoParamCheckBox(inSock);
                    pMiniLayout->addItem(pSocketCheckbox);

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
                        updateSocketDeflValue(nodeid, inSock, inSocket, bChecked);
                    });
                    socket_ctrl.socket_control = pSocketCheckbox;
                }
                else if (ctrl == CONTROL_MULTILINE_STRING)
                {
                    //todo
                }
                else if (ctrl == CONTROL_HEATMAP)
                {
                    //todo
                }
                else if (ctrl == CONTROL_DICTKEY)
                {
                    pSocketItem->setTextInteractionFlags(Qt::TextEditorInteraction);

                    QTextFrame* frame = pSocketItem->document()->rootFrame();
                    QTextFrameFormat format = frame->frameFormat();
                    format.setBackground(QColor(37, 37, 37));
                    frame->setFrameFormat(format);

                    connect(pSocketItem, &ZenoTextLayoutItem::contentsChanged, this, [=](QString oldText, QString newText) {
                        IGraphsModel *pGraphsModel = zenoApp->graphsManagment()->currentModel();
                        ZASSERT_EXIT(pGraphsModel);
                        SOCKET_UPDATE_INFO info;
                        info.bInput = true;
                        info.newInfo.name = newText;
                        info.oldInfo.name = oldText;
                        info.updateWay = SOCKET_UPDATE_NAME;
                        bool ret = pGraphsModel->updateSocketNameNotDesc(m_index.data(ROLE_OBJID).toString(), info, m_subGpIndex, true);
                        if (!ret) {
                            //todo: error hint.
                            pSocketItem->blockSignals(true);
                            pSocketItem->setText(oldText);
                            pSocketItem->blockSignals(false);
                        }
                    });
                    socket_ctrl.socket_control = nullptr;
                }
                else if (ctrl == CONTROL_VEC3F)
                {
                    QVector<qreal> vec = inSocket.info.defaultValue.value<QVector<qreal>>();
                    ZenoVecEditWidget *pVecEditor = new ZenoVecEditWidget(vec);
                    pMiniLayout->addItem(pVecEditor);
                    connect(pVecEditor, &ZenoVecEditWidget::editingFinished, this, [=]() {
                        QVector<qreal> vec = pVecEditor->vec();
                        const QVariant &newValue = QVariant::fromValue(vec);
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

                    connect(pComboBox, &ZenoParamComboBox::textActivated, this, [=](const QString &textValue) {
                        QString oldValue = pComboBox->text();
                        updateSocketDeflValue(nodeid, inSock, inSocket, textValue);
                    });
                    socket_ctrl.socket_control = pComboBox;
                }
                else if (ctrl == CONTROL_CURVE)
                {
                    ZenoParamPushButton *pEditBtn = new ZenoParamPushButton("Edit", -1, QSizePolicy::Expanding);
                    pMiniLayout->addItem(pEditBtn);
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
                            updateSocketDeflValue(nodeid, inSock, inSocket, newValue);
                        });
                    });
                    socket_ctrl.socket_control = pEditBtn;
                }

                socket_ctrl.socket = socket;
                socket_ctrl.socket_text = pSocketItem;
                m_inSockets.insert(inSock, socket_ctrl);
                m_pInSocketsLayout->addItem(pMiniLayout);
                updateWhole();
            }
            else
            {
                //update value directly.
                switch (inSocket.info.control)
                {
                    case CONTROL_STRING:
                    case CONTROL_INT:
                    case CONTROL_FLOAT:{
                        ZenoParamLineEdit *plineEdit =
                            qobject_cast<ZenoParamLineEdit *>(m_inSockets[inSocket.info.name].socket_control);
                        if (plineEdit) {
                            plineEdit->setText(inSocket.info.defaultValue.toString());
                        }
                        break;
                    }
                    case CONTROL_BOOL: {
                        ZenoParamCheckBox* pSocketCheckbox =
                            qobject_cast<ZenoParamCheckBox *>(m_inSockets[inSocket.info.name].socket_control);
                        if (pSocketCheckbox) {
                            bool bChecked = inSocket.info.defaultValue.toBool();
                            pSocketCheckbox->setCheckState(bChecked ? Qt::Checked : Qt::Unchecked);
                        }
                        break;
                    }
                    case CONTROL_VEC3F: {
                        ZenoVecEditWidget *pVecEdit =
                            qobject_cast<ZenoVecEditWidget *>(m_inSockets[inSocket.info.name].socket_control);
                        if (pVecEdit) {
                            const QVector<qreal> &vec = inSocket.info.defaultValue.value<QVector<qreal>>();
                            pVecEdit->setVec(vec);
                        }
                        break;
                    }
                    case CONTROL_ENUM: {
                        ZenoParamComboBox *pComboBox =
                            qobject_cast<ZenoParamComboBox *>(m_inSockets[inSocket.info.name].socket_control);
                        if (pComboBox) {
                            pComboBox->setText(inSocket.info.defaultValue.toString());
                        }
                        break;
                    }
                }
            }
        }
        //remove all keys which don't exist in model.
        for (auto name : m_inSockets.keys())
        {
            if (inputs.find(name) == inputs.end())
            {
                const _socket_ctrl &sock = m_inSockets[name];
                //ugly...
                auto socketlayout = sock.socket_text->parentLayoutItem();
                m_pInSocketsLayout->removeItem(socketlayout);
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

        QStringList coreKeys = outputs.keys();
        QStringList uiKeys = m_outSockets.keys();
        QSet<QString> coreNewSet = coreKeys.toSet().subtract(uiKeys.toSet());
        QSet<QString> uiNewSet = uiKeys.toSet().subtract(coreKeys.toSet());
        //detecting the rename case for "MakeDict".
        if (coreNewSet.size() == 1 && uiNewSet.size() == 1 && m_index.data(ROLE_OBJNAME) == "ExtractDict") {
            //rename.
            QString oldName = *uiNewSet.begin();
            QString newName = *coreNewSet.begin();
            ZASSERT_EXIT(oldName != newName);

            _socket_ctrl ctrl = m_outSockets[oldName];
            ctrl.socket_text->setText(newName);

            //have to reset the text format, which is trivial.
            QTextFrame *frame = ctrl.socket_text->document()->rootFrame();
            QTextFrameFormat format = frame->frameFormat();
            format.setBackground(QColor(37, 37, 37));
            frame->setFrameFormat(format);

            m_outSockets[newName] = ctrl;
            m_outSockets.remove(oldName);

            updateWhole();
            return;
        }

        for (OUTPUT_SOCKET outSocket : outputs)
        {
            const QString& outSock = outSocket.info.name;
            if (m_outSockets.find(outSock) == m_outSockets.end())
            {
                _socket_ctrl sock;
                sock.socket = new ZenoSocketItem(m_renderParams.socket, m_renderParams.szSocket, this);
                sock.socket->setIsInput(false);
                sock.socket_text = new ZenoTextLayoutItem(outSock, m_renderParams.socketFont, m_renderParams.socketClr.color());
                sock.socket_text->setRight(true);

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
                m_pOutSocketsLayout->addItem(pMiniLayout);
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
        }
	}
	else
	{
        if (m_outSockets.find(sockName) != m_outSockets.end())
        {
            m_outSockets[sockName].socket->toggle(bAdded);
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
    onSocketsUpdate(true);
    onSocketsUpdate(false);
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

void ZenoNode::markError(bool isError)
{
    m_bError = isError;
    if (m_bError)
        m_headerWidget->setColors(false, QColor(200, 84, 79), QColor(), QColor());
    else
        m_headerWidget->setColors(false, QColor(83, 96, 147), QColor(), QColor());
    update();
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
