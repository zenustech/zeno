#include "zenoproppanel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/include/igraphsmodel.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/comctrl/zveceditor.h>
#include <zenoui/util/uihelper.h>
#include <zenoui/comctrl/zexpandablesection.h>
#include <zenoui/comctrl/zlinewidget.h>
#include "util/log.h"


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
{
	QVBoxLayout* pVLayout = new QVBoxLayout;
	pVLayout->setContentsMargins(QMargins(0, 0, 0, 0));
	setLayout(pVLayout);
	setFocusPolicy(Qt::ClickFocus);

	QPalette palette = this->palette();
	palette.setBrush(QPalette::Window, QColor(37, 37, 38));
	setPalette(palette);
	setAutoFillBackground(true);

	setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
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
	QHBoxLayout* pTitleLayout = new QHBoxLayout;
	pTitleLayout->setContentsMargins(15, 15, 15, 15);
	QLabel* pLabel = new QLabel(m_idx.data(ROLE_OBJNAME).toString());
	pLabel->setProperty("cssClass", "proppanel-nodename");
	pTitleLayout->addWidget(pLabel);
	pTitleLayout->addStretch();
	QLabel* pWiki = new QLabel(tr("Wiki"));
	pWiki->setProperty("cssClass", "proppanel");
	pTitleLayout->addWidget(pWiki);

	pMainLayout->addLayout(pTitleLayout);

	auto box = inputsBox(pModel, subgIdx, nodes);
	if (box)
	{
		pMainLayout->addWidget(new ZLineWidget(true, QColor(37, 37, 37)));
		pMainLayout->addWidget(box);
	}

	box = paramsBox(pModel, subgIdx, nodes);
	if (box)
	{
		pMainLayout->addWidget(new ZLineWidget(true, QColor(37, 37, 37)));
		pMainLayout->addWidget(box);
	}

	pMainLayout->addStretch();
	pMainLayout->setSpacing(0);

	update();
}

ZExpandableSection* ZenoPropPanel::paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	PARAMS_INFO params = pModel->data2(subgIdx, nodes[0], ROLE_PARAMETERS).value<PARAMS_INFO>();
	if (params.isEmpty())
		return nullptr;

	ZExpandableSection* pParamsBox = new ZExpandableSection("NODE PARAMETERS");
	QGridLayout* pLayout = new QGridLayout;
	pLayout->setContentsMargins(0, 15, 0, 15);
	pLayout->setColumnStretch(0, 1);
	pLayout->setColumnStretch(1, 3);
	pLayout->setSpacing(5);

	int r = 0;
	for (auto paramName : params.keys())
	{
		const PARAM_INFO& param = params[paramName];
		if (param.control == CONTROL_NONE)
			continue;

		QLabel* pNameItem = new QLabel(paramName);
		pNameItem->setProperty("cssClass", "proppanel-itemname");
		pLayout->addWidget(pNameItem, r, 0, Qt::AlignLeft);

		switch (param.control)
		{
			case CONTROL_STRING:
			case CONTROL_INT:
			case CONTROL_FLOAT:
			{
				QLineEdit* pLineEdit = new QLineEdit(param.value.toString());
				pLineEdit->setProperty("cssClass", "proppanel");
				pLineEdit->setObjectName(paramName);
				pLineEdit->setProperty("control", param.control);
				connect(pLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onParamEditFinish);

				pLayout->addWidget(pLineEdit, r++, 1);
				break;
			}
			case CONTROL_BOOL:
			{
				ZCheckBoxBar *pCheckbox = new ZCheckBoxBar;
				pCheckbox->setObjectName(paramName);
				pCheckbox->setCheckState(param.value.toBool()?Qt::Checked:Qt::Unchecked);
				connect(pCheckbox, &ZCheckBoxBar::stateChanged, this, &ZenoPropPanel::onParamEditFinish);

				pLayout->addWidget(pCheckbox, r++, 1);
				break;
			}
			case CONTROL_ENUM:
			{
				QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
				QComboBox* pComboBox = new QComboBox;
				pComboBox->setProperty("cssClass", "proppanel");
				pComboBox->addItems(items);
				pComboBox->setItemDelegate(new ZComboBoxItemDelegate(pComboBox));
				pComboBox->setObjectName(paramName);
				pComboBox->setProperty("control", param.control);
				pComboBox->setCurrentText(param.value.toString());

				//todo: unify
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
				connect(pComboBox, &QComboBox::textActivated, this, &ZenoPropPanel::onParamEditFinish);
#else
				connect(pComboBox, SIGNAL(activated(const QString&)), this, SLOT(onParamEditFinish()));
#endif

				pLayout->addWidget(pComboBox, r++, 1);
				break;
			}
			case CONTROL_READPATH:
			{
				QLineEdit* pathLineEdit = new QLineEdit(param.value.toString());
				pathLineEdit->setProperty("cssClass", "proppanel");
				pathLineEdit->setObjectName(paramName);
				pathLineEdit->setProperty("control", param.control);
				pLayout->addWidget(pathLineEdit, r, 1);
				connect(pathLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onParamEditFinish);

				ZIconLabel* openBtn = new ZIconLabel;
				openBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(28, 28)), ":/icons/ic_openfile.svg", ":/icons/ic_openfile-on.svg", ":/icons/ic_openfile-on.svg");
				pLayout->addWidget(openBtn, r++, 2);
				break;
			}
			case CONTROL_WRITEPATH:
			{
				QLineEdit* pathLineEdit = new QLineEdit(param.value.toString());
				pathLineEdit->setProperty("cssClass", "proppanel");
				pathLineEdit->setObjectName(paramName);
				pathLineEdit->setProperty("control", param.control);
				pLayout->addWidget(pathLineEdit, r, 1);
				connect(pathLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onParamEditFinish);

				ZIconLabel* openBtn = new ZIconLabel;
				openBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(28, 28)), ":/icons/ic_openfile.svg", ":/icons/ic_openfile-on.svg", ":/icons/ic_openfile-on.svg");
				pLayout->addWidget(openBtn, r++, 2);
				break;
			}
			case CONTROL_MULTILINE_STRING:
			{
				QTextEdit* pTextEdit = new QTextEdit;
				pTextEdit->setFrameShape(QFrame::NoFrame);
				pTextEdit->setProperty("cssClass", "proppanel");
				pTextEdit->setObjectName(paramName);
				pTextEdit->setProperty("control", param.control);
				pTextEdit->setFont(QFont("HarmonyOS Sans", 12));

				//todo: ztextedit impl.

				QTextCharFormat format;
				QFont font("HarmonyOS Sans", 12);
				format.setFont(font);
				pTextEdit->setCurrentFont(font);
				pTextEdit->setText(param.value.toString());

				QPalette pal = pTextEdit->palette();
				pal.setColor(QPalette::Base, QColor(37, 37, 37));
				pTextEdit->setPalette(pal);

				pLayout->addWidget(pTextEdit, r++, 1);
				break;
			}
			case CONTROL_HEATMAP:
			{
				QPushButton* pBtn = new QPushButton("Edit Heatmap");
				pBtn->setObjectName("grayButton");
                pBtn->setProperty("cssClass", "grayButton");
				pLayout->addWidget(pBtn, r++, 1);
				break;
			}
			case CONTROL_CURVE:
            {
				QPushButton* pBtn = new QPushButton("Edit Curve");
				pBtn->setObjectName("grayButton");
                pBtn->setProperty("cssClass", "grayButton");
				pLayout->addWidget(pBtn, r++, 1);
                break;
            }
			default:
			{
				break;
			}
		}
	}

	pParamsBox->setContentLayout(pLayout);
	return pParamsBox;
}

ZExpandableSection* ZenoPropPanel::inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	INPUT_SOCKETS inputs = pModel->data2(subgIdx, nodes[0], ROLE_INPUTS).value<INPUT_SOCKETS>();
	if (inputs.keys().isEmpty())
		return nullptr;

	QGridLayout* pLayout = new QGridLayout;
	pLayout->setContentsMargins(0, 15, 0, 15);
	pLayout->setSpacing(5);

	int r = 0;
	for (QString inputSock : inputs.keys())
	{
		ZASSERT_EXIT(inputs.find(inputSock) != inputs.end(), nullptr);
		INPUT_SOCKET input = inputs[inputSock];

		switch (input.info.control)
		{
			case CONTROL_STRING:
			case CONTROL_FLOAT:
			case CONTROL_INT:
			{
				QLabel* pNameItem = new QLabel(inputSock);
				pNameItem->setProperty("cssClass", "proppanel");
				pLayout->addWidget(pNameItem, r, 0, Qt::AlignLeft);

				QLineEdit* pLineEdit = new QLineEdit(UiHelper::variantToString(input.info.defaultValue));
				pLineEdit->setProperty("cssClass", "proppanel");
				pLineEdit->setObjectName(inputSock);
				pLineEdit->setProperty("control", input.info.control);
				connect(pLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onInputEditFinish);

				pLayout->addWidget(pLineEdit, r++, 1);
				break;
			}
			case CONTROL_BOOL:
			{
				QLabel *pNameItem = new QLabel(inputSock);
				pNameItem->setProperty("cssClass", "proppanel");
				pLayout->addWidget(pNameItem, r, 0, Qt::AlignLeft);

				ZCheckBoxBar *pCheckbox = new ZCheckBoxBar;
				pCheckbox->setObjectName(inputSock);
				pCheckbox->setCheckState(input.info.defaultValue.toBool() ? Qt::Checked : Qt::Unchecked);
				connect(pCheckbox, &ZCheckBoxBar::stateChanged, this, &ZenoPropPanel::onInputEditFinish);
				pLayout->addWidget(pCheckbox, r++, 1);
				break;
			}
			case CONTROL_VEC3F:
			{
				QLabel* pNameItem = new QLabel(inputSock);
				pNameItem->setProperty("cssClass", "proppanel");
				pLayout->addWidget(pNameItem, r, 0, Qt::AlignLeft);

				QVector<qreal> vec = input.info.defaultValue.value<QVector<qreal>>();
				ZVecEditor* pVecEdit = new ZVecEditor(vec, true, 3, "proppanel");
				pVecEdit->setObjectName(inputSock);
				connect(pVecEdit, &ZVecEditor::editingFinished, this, &ZenoPropPanel::onInputEditFinish);

				pLayout->addWidget(pVecEdit, r++, 1);
				break;
			}
			case CONTROL_ENUM:
			{
				QLabel* pNameItem = new QLabel(inputSock);
				pNameItem->setProperty("cssClass", "proppanel");
				pLayout->addWidget(pNameItem, r, 0, Qt::AlignLeft);

				QString descStr = input.info.type;
				QStringList items = descStr.mid(QString("enum ").length()).split(QRegExp("\\s+"));

				ZComboBox *pComboBox = new ZComboBox;
				pComboBox->addItems(items);
				pComboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
				pComboBox->setItemDelegate(new ZComboBoxItemDelegate(pComboBox));
				pComboBox->setObjectName(inputSock);
                connect(pComboBox, &QComboBox::currentTextChanged, this, &ZenoPropPanel::onInputEditFinish);

				QString val = input.info.defaultValue.toString();
				if (items.indexOf(val) != -1)
				{
					pComboBox->setCurrentText(val);
				}
				pLayout->addWidget(pComboBox, r++, 1);
				break;
			}
		}
	}

	if (pLayout->count() == 0)
	{
		delete pLayout;
		return nullptr;
	}

	ZExpandableSection* pInputsBox = new ZExpandableSection("SOCKET IN");
	pInputsBox->setContentLayout(pLayout);
	return pInputsBox;
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
	
	if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(pSender))
	{
		QString textValue = pLineEdit->text();
		info.newValue = UiHelper::_parseDefaultValue(textValue, inSocket.info.type);
	}
	else if (ZVecEditor* pVecEdit = qobject_cast<ZVecEditor*>(pSender))
	{
		QVector<qreal> vec = pVecEdit->vec();
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

	if (info.oldValue != info.newValue)
	{
		IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
		ZASSERT_EXIT(pGraphsModel);
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

	if (QLineEdit* pLineEdit = qobject_cast<QLineEdit*>(pSender))
	{
		textValue = pLineEdit->text();
	}
	else if (QComboBox* pCombobox = qobject_cast<QComboBox*>(pSender))
	{
		textValue = pCombobox->currentText();
	}
	else if (QTextEdit* pTextEdit = qobject_cast<QTextEdit*>(pSender))
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

void ZenoPropPanel::onDataChanged(const QModelIndex& subGpIdx, const QModelIndex& idx, int role)
{
	//may be called frequently
	if (m_subgIdx != subGpIdx || m_idx != idx)
		return;

	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	if (!pModel)
		return;

	if (role == ROLE_PARAMETERS)
	{
		const PARAMS_INFO& params = pModel->data2(m_subgIdx, m_idx, role).value<PARAMS_INFO>();
		for (PARAM_INFO param : params)
		{
			switch (param.control)
			{
				case CONTROL_STRING:
				case CONTROL_INT:
				case CONTROL_FLOAT:
				case CONTROL_READPATH:
				case CONTROL_WRITEPATH:
				{
					//update lineedit
					auto lst = findChildren<QLineEdit*>(param.name, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QLineEdit* pEdit = lst[0];
						pEdit->setText(param.value.toString());
					}
					break;
				}
				case CONTROL_BOOL:
				{
					//update lineedit
					auto lst = findChildren<ZCheckBoxBar*>(param.name, Qt::FindChildrenRecursively);
					if (lst.size() == 1) {
						ZCheckBoxBar *pEdit = lst[0];
						pEdit->setCheckState(param.value.toBool() ? Qt::Checked : Qt::Unchecked);
					}
					break;
				}
				case CONTROL_ENUM:
				{
					auto lst = findChildren<QComboBox*>(param.name, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QComboBox* pCombo = lst[0];
						pCombo->setCurrentText(param.value.toString());
					}
					break;
				}
				case CONTROL_MULTILINE_STRING:
				{
					auto lst = findChildren<QTextEdit*>(param.name, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QTextEdit* pTextEdit = lst[0];
						pTextEdit->setText(param.value.toString());
					}
					break;
				}
				case CONTROL_HEATMAP:
                case CONTROL_CURVE:  //TODO(bate): find the QPushButton
				{
					//update lineedit
					auto lst = findChildren<QLineEdit*>(param.name, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QLineEdit* pEdit = lst[0];
						pEdit->setText(param.value.toString());
					}
					break;
				}
			}
		}
	}
	else if (role == ROLE_INPUTS)
	{
		const INPUT_SOCKETS& inSocks = pModel->data2(m_subgIdx, m_idx, role).value<INPUT_SOCKETS>();
		for (QString inSock : inSocks.keys())
		{
			const INPUT_SOCKET& inSocket = inSocks[inSock];
			switch (inSocket.info.control)
			{
				case CONTROL_STRING:
				case CONTROL_INT:
				case CONTROL_FLOAT:
				case CONTROL_READPATH:
				case CONTROL_WRITEPATH:
				{
					//update lineedit
					auto lst = findChildren<QLineEdit*>(inSock, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QLineEdit* pEdit = lst[0];
						pEdit->setText(inSocket.info.defaultValue.toString());
					}
					break;
				}
				case CONTROL_BOOL:
				{
					auto lst = findChildren<ZCheckBoxBar*>(inSock, Qt::FindChildrenRecursively);
					if (lst.size() == 1) {
						ZCheckBoxBar *pEdit = lst[0];
						pEdit->setCheckState(inSocket.info.defaultValue.toBool() ? Qt::Checked : Qt::Unchecked);
					}
					break;
				}
				case CONTROL_VEC3F:
				{
					auto lst = findChildren<ZVecEditor*>(inSock, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						ZVecEditor* pEdit = lst[0];
						pEdit->onValueChanged(inSocket.info.defaultValue.value<QVector<qreal>>());
					}
					break;
				}
				case CONTROL_ENUM:
				{
					auto lst = findChildren<QComboBox*>(inSock, Qt::FindChildrenRecursively);
					if (lst.size() == 1)
					{
						QComboBox* pComboBox = lst[0];
						pComboBox->setCurrentText(inSocket.info.defaultValue.toString());
					}
					break;
				}
			}
		}
	}
}
