#include "zenoproppanel.h"
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include "graphsmanagment.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/model/curvemodel.h>
#include <zenoui/model/variantptr.h>
#include <zenoui/include/igraphsmodel.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/comctrl/zexpandablesection.h>
#include <zenoui/comctrl/zlinewidget.h>
#include <zenoui/comctrl/zlineedit.h>
#include <zenoui/comctrl/ztextedit.h>
#include "util/log.h"
#include "util/apphelper.h"
#include "curvemap/curveutil.h"
#include "curvemap/zcurvemapeditor.h"
#include "panel/zenoheatmapeditor.h"


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
	, m_bReentry(false)
{
	QVBoxLayout* pVLayout = new QVBoxLayout;
	pVLayout->setContentsMargins(QMargins(0, 0, 0, 0));
	setLayout(pVLayout);
	setFocusPolicy(Qt::ClickFocus);

	QPalette palette = this->palette();
	palette.setBrush(QPalette::Window, QColor(44, 51, 58));
	setPalette(palette);
	setAutoFillBackground(true);
}

ZenoPropPanel::~ZenoPropPanel()
{
}

QSize ZenoPropPanel::sizeHint() const
{
    QSize sz = QWidget::sizeHint();
    return sz;
}

QSize ZenoPropPanel::minimumSizeHint() const
{
    QSize sz = QWidget::minimumSizeHint();
    return sz;
}

void ZenoPropPanel::clearLayout()
{
    setUpdatesEnabled(false);
	qDeleteAll(findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly));
	QVBoxLayout* pMainLayout = qobject_cast<QVBoxLayout*>(this->layout());
	while (pMainLayout->count() > 0)
	{
		QLayoutItem* pItem = pMainLayout->itemAt(pMainLayout->count() - 1);
		pMainLayout->removeItem(pItem);
	}
	setUpdatesEnabled(true);
	update();
}

void ZenoPropPanel::reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
	//if (!nodes.isEmpty() && nodes[0] == m_idx)
	//	return;

    clearLayout();
    QVBoxLayout *pMainLayout = qobject_cast<QVBoxLayout *>(this->layout());

	if (!pModel || !select || nodes.isEmpty())
	{
		update();
		return;
	}

    connect(pModel, &IGraphsModel::_dataChanged, this, &ZenoPropPanel::onDataChanged);
    connect(pModel, &IGraphsModel::_rowsRemoved, this, [=]() {
		clearLayout();
    });
    connect(pModel, &IGraphsModel::modelClear, this, [=]() {
		clearLayout();
    });

	m_subgIdx = subgIdx;
	m_idx = nodes[0];

	//title
	//QHBoxLayout* pTitleLayout = new QHBoxLayout;
	//pTitleLayout->setContentsMargins(15, 15, 15, 15);
	//QLabel* pLabel = new QLabel(m_idx.data(ROLE_OBJNAME).toString());
	//pLabel->setProperty("cssClass", "proppanel-nodename");
	//pTitleLayout->addWidget(pLabel);
	//pTitleLayout->addStretch();
	//QLabel* pWiki = new QLabel(tr("Wiki"));
	//pWiki->setProperty("cssClass", "proppanel");
	//pTitleLayout->addWidget(pWiki);

	//pMainLayout->addLayout(pTitleLayout);

	auto box = inputsBox(pModel, subgIdx, nodes);
	if (box)
	{
		pMainLayout->addWidget(box);
	}

	box = paramsBox(pModel, subgIdx, nodes);
	if (box)
	{
		pMainLayout->addWidget(box);
	}

	pMainLayout->addStretch();
	pMainLayout->setSpacing(0);

	onInputsCheckUpdate();
	onParamsCheckUpdate();

	update();
}

ZExpandableSection* ZenoPropPanel::paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	ZASSERT_EXIT(m_idx.isValid(), nullptr);

	PARAMS_INFO params = m_idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	if (params.isEmpty())
		return nullptr;

	ZExpandableSection* pParamsBox = new ZExpandableSection(tr("NODE PARAMETERS"));
	pParamsBox->setObjectName(tr("NODE PARAMETERS"));
	QGridLayout* pLayout = new QGridLayout;
	pLayout->setContentsMargins(10, 15, 0, 15);
	pLayout->setColumnStretch(0, 1);
	pLayout->setColumnStretch(1, 3);
	pLayout->setSpacing(10);
	pParamsBox->setContentLayout(pLayout);
	return pParamsBox;
}

ZExpandableSection* ZenoPropPanel::inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	ZASSERT_EXIT(m_idx.isValid(), nullptr);

    INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    if (inputs.isEmpty())
        return nullptr;

	const QString& groupName = tr("SOCKET IN");
	ZExpandableSection* pInputsBox = new ZExpandableSection(groupName);
	pInputsBox->setObjectName(groupName);
	QGridLayout* pLayout = new QGridLayout;
    pLayout->setContentsMargins(10, 15, 0, 15);
    pLayout->setColumnStretch(0, 1);
    pLayout->setColumnStretch(1, 3);
    pLayout->setSpacing(10);
	pInputsBox->setContentLayout(pLayout);
	return pInputsBox;
}

QWidget* ZenoPropPanel::initControl(CONTROL_DATA ctrlData)
{
	PARAM_CONTROL ctrl = ctrlData.ctrl;
	const QString& name = ctrlData.name;
	const QVariant& value = ctrlData.value;
	const QString& typeDesc = ctrlData.typeDesc;

    switch (ctrl)
    {
		case CONTROL_STRING:
		case CONTROL_FLOAT:
		case CONTROL_INT:
		{
			ZLineEdit* pLineEdit = new ZLineEdit(UiHelper::variantToString(value));
			pLineEdit->setProperty("cssClass", "proppanel");
			pLineEdit->setNumSlider(UiHelper::getSlideStep(name, ctrl));
			pLineEdit->setObjectName(name);
			//todo: validator.
			connect(pLineEdit, &ZLineEdit::editingFinished, this, ctrlData.fSlot);
			return pLineEdit;
		}
		case CONTROL_BOOL:
		{
			ZCheckBoxBar* pCheckbox = new ZCheckBoxBar;
			pCheckbox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
			pCheckbox->setObjectName(name);
			connect(pCheckbox, &ZCheckBoxBar::stateChanged, this, ctrlData.fSlot);
			return pCheckbox;
		}
		case CONTROL_VEC:
		{
			UI_VECTYPE vec = value.value<UI_VECTYPE>();
			int dim = -1;
			bool bFloat = false;
			UiHelper::parseVecType(typeDesc, dim, bFloat);

			ZVecEditor* pVecEdit = new ZVecEditor(vec, bFloat, 3, "proppanel");
			pVecEdit->setObjectName(name);
			connect(pVecEdit, &ZVecEditor::editingFinished, this, ctrlData.fSlot);
			return pVecEdit;
		}
		case CONTROL_ENUM:
		{
			QStringList items = typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
			QComboBox* pComboBox = new QComboBox;
			pComboBox->setProperty("cssClass", "proppanel");
			pComboBox->setObjectName(name);
			pComboBox->addItems(items);
			pComboBox->setItemDelegate(new ZComboBoxItemDelegate(pComboBox));
			pComboBox->setObjectName(name);
			pComboBox->setProperty("control", ctrl);
			pComboBox->setCurrentText(value.toString());

			//todo: unify
	#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
			connect(pComboBox, &QComboBox::textActivated, this, ctrlData.fSlot);
	#else
			connect(pComboBox, &QComboBox::activated, this, ctrlData.fSlot);
	#endif
			return pComboBox;
		}
		case CONTROL_READPATH:
		case CONTROL_WRITEPATH:
		{
			ZLineEdit* pathLineEdit = new ZLineEdit(value.toString());
			pathLineEdit->setIcons(":/icons/ic_openfile.svg", ":/icons/ic_openfile-on.svg");
			pathLineEdit->setProperty("cssClass", "proppanel");
			pathLineEdit->setObjectName(name);
			pathLineEdit->setProperty("control", ctrl);
			pathLineEdit->setFocusPolicy(Qt::ClickFocus);
			connect(pathLineEdit, &ZLineEdit::btnClicked, this, [=]() {
				bool bRead = ctrl == CONTROL_READPATH;
				QString path;
				DlgInEventLoopScope;
				if (bRead) {
					path = QFileDialog::getOpenFileName(nullptr, tr("File to Open"), "", tr("All Files(*);;"));
				}
				else {
					path = QFileDialog::getSaveFileName(nullptr, tr("Path to Save"), "", tr("All Files(*);;"));
				}
				pathLineEdit->setText(path);
				emit pathLineEdit->textEditFinished();
				pathLineEdit->clearFocus();
			});
			connect(pathLineEdit, &ZLineEdit::textEditFinished, this, ctrlData.fSlot);
			return pathLineEdit;
		}
		case CONTROL_MULTILINE_STRING:
		{
			ZTextEdit* pTextEdit = new ZTextEdit;
			pTextEdit->setFrameShape(QFrame::NoFrame);
			pTextEdit->setProperty("cssClass", "proppanel");
			pTextEdit->setObjectName(name);
			pTextEdit->setProperty("control", ctrl);
			pTextEdit->setFont(QFont("HarmonyOS Sans", 12));

			QTextCharFormat format;
			QFont font("HarmonyOS Sans", 12);
			format.setFont(font);
			pTextEdit->setCurrentFont(font);
			pTextEdit->setText(value.toString());

			QPalette pal = pTextEdit->palette();
			pal.setColor(QPalette::Base, QColor(37, 37, 37));
			pTextEdit->setPalette(pal);

			connect(pTextEdit, &ZTextEdit::editFinished, this, ctrlData.fSlot);
			return pTextEdit;
		}
		case CONTROL_COLOR:
		{
			QPushButton* pBtn = new QPushButton("Edit Heatmap");
			pBtn->setObjectName(name);
			pBtn->setProperty("cssClass", "grayButton");
			connect(pBtn, &QPushButton::clicked, this, ctrlData.fSlot);
			return pBtn;
		}
		case CONTROL_CURVE:
		{
			QPushButton* pBtn = new QPushButton("Edit Curve");
			pBtn->setObjectName(name);
			pBtn->setProperty("cssClass", "grayButton");
			connect(pBtn, &QPushButton::clicked, this, ctrlData.fSlot);
			return pBtn;
		}
		default:
		{
			return nullptr;
		}
    }
}

void ZenoPropPanel::onInputsCheckUpdate()
{
	ZASSERT_EXIT(m_idx.isValid());
    INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    QMap<QString, CONTROL_DATA> ctrls;
    for (const QString& inSock : inputs.keys())
    {
        const INPUT_SOCKET& inSocket = inputs[inSock];
        CONTROL_DATA ctrl;
        ctrl.ctrl = inSocket.info.control;
        ctrl.name = inSock;
        ctrl.typeDesc = inSocket.info.type;
        ctrl.value = inSocket.info.defaultValue;
        if (ctrl.ctrl == CONTROL_COLOR)
        {
            ctrl.fSlot = [this]() {
                QPushButton* pSender = qobject_cast<QPushButton*>(sender());
                QString inSock = pSender->objectName();
				onInputColorEdited(inSock);
            };
        }
        else if (ctrl.ctrl == CONTROL_CURVE)
        {
            ctrl.fSlot = [this]() {
                QPushButton* pSender = qobject_cast<QPushButton*>(sender());
                QString paramName = pSender->objectName();
                onCurveModelEdit(true, paramName);
            };
        }
		else
		{
			ctrl.fSlot = std::bind(&ZenoPropPanel::onInputEditFinish, this);
		}
        ctrls.insert(inSock, ctrl);
    }
    onGroupCheckUpdated(tr("SOCKET IN"), ctrls);
}

void ZenoPropPanel::onParamsCheckUpdate()
{
	ZASSERT_EXIT(m_idx.isValid());
	PARAMS_INFO params = m_idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	QMap<QString, CONTROL_DATA> ctrls;
	for (const QString& name : params.keys())
	{
		const PARAM_INFO& param = params[name];
        CONTROL_DATA ctrl;
        ctrl.ctrl = param.control;
        ctrl.name = name;
        ctrl.typeDesc = param.typeDesc;
        ctrl.value = param.value;
		if (ctrl.ctrl == CONTROL_COLOR)
		{
			ctrl.fSlot = [this]() {
				QPushButton* pSender = qobject_cast<QPushButton*>(sender());
				QString paramName = pSender->objectName();
				onParamColorEdited(paramName);
			};
		}
		else if (ctrl.ctrl == CONTROL_CURVE)
		{
			ctrl.fSlot = [this]() {
				QPushButton* pSender = qobject_cast<QPushButton*>(sender());
				QString paramName = pSender->objectName();
				onCurveModelEdit(false, paramName);
			};
		}
		else
		{
			ctrl.fSlot = std::bind(&ZenoPropPanel::onParamEditFinish, this);
		}
        ctrls.insert(name, ctrl);
	}
	onGroupCheckUpdated(tr("NODE PARAMETERS"), ctrls);
}

void ZenoPropPanel::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
}

void ZenoPropPanel::onInputEditFinish()
{
	QObject* pSender = sender();
	IGraphsModel* model = zenoApp->graphsManagment()->currentModel();
	if (!model)
		return;

	const QString& inSock = pSender->objectName();
	const QString& nodeid = m_idx.data(ROLE_OBJID).toString();
	const INPUT_SOCKETS& inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
	const INPUT_SOCKET& inSocket = inputs[inSock];

	PARAM_UPDATE_INFO info;
	info.name = inSock;
	info.oldValue = inSocket.info.defaultValue;
	
	if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pSender))
	{
		QString textValue = pLineEdit->text();
		info.newValue = UiHelper::_parseDefaultValue(textValue, inSocket.info.type);
	}
	else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pSender))
	{
		UI_VECTYPE vec = pVecEdit->vec();
		info.newValue = QVariant::fromValue(vec);
	}
	else if (QComboBox* pComboBox = qobject_cast<QComboBox*>(pSender))
	{
		info.newValue = pComboBox->currentText();
	}
	else if (ZCheckBoxBar* pCheckbox = qobject_cast<ZCheckBoxBar*>(pSender))
	{
		info.newValue = pCheckbox->checkState() == Qt::Checked;
	}
	else if (ZTextEdit* pTextEdit = qobject_cast<ZTextEdit*>(pSender))
	{
		info.newValue = pTextEdit->toPlainText();
	}

	if (info.oldValue != info.newValue)
	{
		IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
		ZASSERT_EXIT(pGraphsModel);
		zeno::scope_exit se([this]() { m_bReentry = false; });
		m_bReentry = true;
		pGraphsModel->updateSocketDefl(nodeid, info, m_subgIdx, true);
	}
}

void ZenoPropPanel::onParamEditFinish()
{
	QObject* pSender = sender();
	IGraphsModel* model = zenoApp->graphsManagment()->currentModel();
	if (!model)
		return;

	const QString& paramName = pSender->objectName();
	PARAM_CONTROL ctrl = (PARAM_CONTROL)pSender->property("control").toInt();
	const QString& nodeid = m_idx.data(ROLE_OBJID).toString();
	QString textValue;

	if (ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pSender))
	{
		textValue = pLineEdit->text();
	}
	else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(pSender))
	{
		textValue = pCombobox->currentText();
	}
    else if (ZTextEdit* pTextEdit = qobject_cast<ZTextEdit*>(pSender))
    {
        textValue = pTextEdit->toPlainText();
    }
	else if (ZCheckBoxBar *pCheckbox = qobject_cast<ZCheckBoxBar *>(pSender))
	{
		PARAM_UPDATE_INFO info;
		info.oldValue = model->getParamValue(nodeid, paramName, m_subgIdx);
		info.newValue = pCheckbox->checkState() == Qt::Checked;
		info.name = paramName;
		if (info.newValue != info.oldValue)
		{
			model->updateParamInfo(nodeid, info, m_subgIdx, true);
		}
		return;
	}
	else
	{
		return;
	}

	PARAM_UPDATE_INFO info;
	info.oldValue = model->getParamValue(nodeid, paramName, m_subgIdx);
	info.newValue = UiHelper::parseTextValue(ctrl, textValue);;
	info.name = paramName;
	model->updateParamInfo(nodeid, info, m_subgIdx, true);
}

void ZenoPropPanel::onCurveModelEdit(bool bInputSock, const QString& name)
{
    IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
    ZASSERT_EXIT(pGraphsModel);
    const QString& nodeid = m_idx.data(ROLE_OBJID).toString();
	QVariant val;
	if (bInputSock) {
		INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
		ZASSERT_EXIT(inputs.find(name) != inputs.end());
		val = inputs[name].info.defaultValue;
	}
	else {
		val = pGraphsModel->getParamValue(nodeid, name, m_subgIdx);
    }
    CurveModel* pModel = QVariantPtr<CurveModel>::asPtr(val);
    ZASSERT_EXIT(pModel);	//the param has been inited as curve model.
    ZCurveMapEditor* pEditor = new ZCurveMapEditor(true);
    pEditor->setAttribute(Qt::WA_DeleteOnClose);
    pEditor->addCurve(pModel);
    pEditor->show();
}

void ZenoPropPanel::onInputColorEdited(const QString& inSock)
{
	INPUT_SOCKETS inputs = m_idx.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
	ZASSERT_EXIT(inputs.find(inSock) != inputs.end());
	INPUT_SOCKET input = inputs[inSock];
	const QString& oldColor = input.info.defaultValue.toString();
    QLinearGradient grad = AppHelper::colorString2Grad(oldColor);

    ZenoHeatMapEditor editor(grad);
    editor.exec();

    QLinearGradient newGrad = editor.colorRamps();
    QString colorText = AppHelper::gradient2colorString(newGrad);
    if (colorText != oldColor)
    {
        PARAM_UPDATE_INFO info;
        info.name = "_RAMPS";
        info.oldValue = oldColor;
        info.newValue = colorText;
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
        ZASSERT_EXIT(pModel);
        zeno::scope_exit se([this]() { m_bReentry = false; });
        m_bReentry = true;
		pModel->updateSocketDefl(m_idx.data(ROLE_OBJID).toString(), info, m_subgIdx, true);
    }
}

void ZenoPropPanel::onParamColorEdited(const QString& paramName)
{
	PARAMS_INFO params = m_idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
	ZASSERT_EXIT(params.find(paramName) != params.end());
    PARAM_INFO& param = params[paramName];
    const QString& oldColor = param.value.toString();
    QLinearGradient grad = AppHelper::colorString2Grad(oldColor);

    ZenoHeatMapEditor editor(grad);
    editor.exec();

    QLinearGradient newGrad = editor.colorRamps();
    QString colorText = AppHelper::gradient2colorString(newGrad);
    if (colorText != oldColor)
    {
        PARAM_UPDATE_INFO info;
        info.name = "_RAMPS";
        info.oldValue = oldColor;
        info.newValue = colorText;
        IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
		ZASSERT_EXIT(pModel);
        zeno::scope_exit se([this]() { m_bReentry = false; });
        m_bReentry = true;
        pModel->updateParamInfo(m_idx.data(ROLE_OBJID).toString(), info, m_subgIdx, true);
    }
}

void ZenoPropPanel::onGroupCheckUpdated(const QString& groupName, const QMap<QString, CONTROL_DATA>& ctrls)
{
	if (ctrls.isEmpty())
		return;

	ZExpandableSection* pExpand = findChild<ZExpandableSection*>(groupName);
	ZASSERT_EXIT(pExpand);

	QGridLayout* pLayout = qobject_cast<QGridLayout*>(pExpand->contentLayout());
	ZASSERT_EXIT(pLayout);
	for (CONTROL_DATA ctrldata : ctrls)
	{
		QLabel* pNameItem = nullptr;
        QWidget* pControl = pExpand->findChild<QWidget*>(ctrldata.name);
		if (!isMatchControl(ctrldata.ctrl, pControl))
		{
			if (pControl)
			{
				//remove the dismatch control
				pLayout->removeWidget(pControl);
				delete pControl;
				pControl = nullptr;
			}

            pControl = initControl(ctrldata);
            if (!pControl)
                continue;

			int n = pLayout->rowCount();
			const QString& lblObjName = "label-" + ctrldata.name;
			if (!pExpand->findChild<QLabel*>(lblObjName))
			{
                pNameItem = new QLabel(ctrldata.name);
				pNameItem->setObjectName(lblObjName);
                pNameItem->setProperty("cssClass", "proppanel");
				pLayout->addWidget(pNameItem, n, 0, Qt::AlignLeft);
			}
            pLayout->addWidget(pControl, n, 1);
		}
		updateControlValue(pControl, ctrldata.ctrl, ctrldata.value);
	}
}

void ZenoPropPanel::onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
	//may be called frequently
	if (m_subgIdx != subGpIdx || m_idx != idx || m_bReentry)
		return;

	QLayout* pLayout = this->layout();
	if (!pLayout || pLayout->isEmpty())
		return;

	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	if (!pModel)
		return;

	if (role == ROLE_PARAMETERS)
	{
		onParamsCheckUpdate();
	}
	else if (role == ROLE_INPUTS)
	{
		onInputsCheckUpdate();
	}
	else
	{
		//other custom ui role.
		//onCustomUiUpdate();
	}
}

bool ZenoPropPanel::isMatchControl(PARAM_CONTROL ctrl, QWidget* pControl)
{
	if (!pControl)
		return false;

	switch (ctrl)
	{
    case CONTROL_STRING:
    case CONTROL_INT:
    case CONTROL_FLOAT:	return qobject_cast<ZLineEdit*>(pControl) != nullptr;
	case CONTROL_READPATH:
	case CONTROL_WRITEPATH: return qobject_cast<ZLineEdit*>(pControl) != nullptr;
	case CONTROL_BOOL:	return qobject_cast<ZCheckBoxBar*>(pControl) != nullptr;
	case CONTROL_VEC:	return qobject_cast<ZVecEditor*>(pControl) != nullptr;
	case CONTROL_ENUM:	return qobject_cast<QComboBox*>(pControl) != nullptr;
	case CONTROL_MULTILINE_STRING:	return qobject_cast<ZTextEdit*>(pControl) != nullptr;
	case CONTROL_CURVE: 
	case CONTROL_COLOR:	return qobject_cast<QPushButton*>(pControl) != nullptr;
	}
}

void ZenoPropPanel::updateControlValue(QWidget* pControl, PARAM_CONTROL ctrl, const QVariant& value)
{
	ZASSERT_EXIT(pControl);
    switch (ctrl)
    {
		case CONTROL_STRING:
		case CONTROL_INT:
		case CONTROL_FLOAT:
		{
			ZLineEdit* pLineEdit = qobject_cast<ZLineEdit*>(pControl);
			pLineEdit->setText(value.toString());
			//todo: validator.
			break;
		}
		case CONTROL_READPATH:
		case CONTROL_WRITEPATH:
		{
			ZLineEdit* pPathEdit = qobject_cast<ZLineEdit*>(pControl);
			pPathEdit->setText(value.toString());
			break;
		}
		case CONTROL_BOOL:
		{
			ZCheckBoxBar* pCheckBox = qobject_cast<ZCheckBoxBar*>(pControl);
			pCheckBox->setCheckState(value.toBool() ? Qt::Checked : Qt::Unchecked);
			break;
		}
		case CONTROL_VEC:
		{
			ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pControl);
			pVecEdit->setVec(value.value<UI_VECTYPE>(), pVecEdit->isFloat());
			break;
		}
		case CONTROL_MULTILINE_STRING:
		{
			ZTextEdit* pTextEdit = qobject_cast<ZTextEdit*>(pControl);
			pTextEdit->setText(value.toString());
			break;
		}
		case CONTROL_ENUM:
		{
			QComboBox* pComboBox = qobject_cast<QComboBox*>(pControl);
			pComboBox->setCurrentText(value.toString());
			break;
		}
		case CONTROL_COLOR:
		case CONTROL_CURVE:
		{
			//just a button, no need to update.
			break;
		}
    }
}
