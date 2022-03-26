#include "zenoproppanel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/include/igraphsmodel.h>
#include <zenoui/comctrl/zcombobox.h>
#include <zenoui/comctrl/zlabel.h>
#include <zenoui/style/zenostyle.h>
#include <zenoui/comctrl/gv/zenoparamwidget.h>
#include <zenoui/util/uihelper.h>


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
{
	QVBoxLayout* pVLayout = new QVBoxLayout;
	pVLayout->setContentsMargins(QMargins(25, 12, 25, 12));
	setLayout(pVLayout);
	setFocusPolicy(Qt::ClickFocus);
}

ZenoPropPanel::~ZenoPropPanel()
{
}

void ZenoPropPanel::reset(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
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

	if (!pModel || !select || nodes.isEmpty())
	{
		update();
		return;
	}

	connect(pModel, &IGraphsModel::_dataChanged, this, &ZenoPropPanel::onDataChanged);

	m_subgIdx = subgIdx;
	m_idx = nodes[0];

	auto box = inputsBox(pModel, subgIdx, nodes);
	if (box)
	{
		pMainLayout->addWidget(box);
		pMainLayout->addSpacing(30);
	}
	pMainLayout->addWidget(paramsBox(pModel, subgIdx, nodes));
	pMainLayout->addStretch();

	update();
}

QGroupBox* ZenoPropPanel::paramsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	PARAMS_INFO params = pModel->data2(subgIdx, nodes[0], ROLE_PARAMETERS).value<PARAMS_INFO>();
	if (params.isEmpty())
		return nullptr;

	QGroupBox* pParamsBox = new QGroupBox;
	QVBoxLayout* pLayout = new QVBoxLayout;
	for (auto paramName : params.keys())
	{
		const PARAM_INFO& param = params[paramName];
		switch (param.control)
		{
		case CONTROL_STRING:
		case CONTROL_INT:
		case CONTROL_FLOAT:
		case CONTROL_BOOL:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(paramName);
			pNameItem->setProperty("cssClass", "proppanel");

			pHLayout->addWidget(pNameItem);

			QLineEdit* pLineEdit = new QLineEdit(param.value.toString());
			pLineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
			pLineEdit->setProperty("cssClass", "proppanel");
			pLineEdit->setObjectName(paramName);
			pLineEdit->setProperty("control", param.control);

			pHLayout->addWidget(pLineEdit);

			connect(pLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onLineEditFinish);
			pLayout->addLayout(pHLayout);
			break;
		}
		case CONTROL_ENUM:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(paramName);
			pNameItem->setProperty("cssClass", "proppanel");
			pHLayout->addWidget(pNameItem);

			QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
			QComboBox* pComboBox = new QComboBox;
			pComboBox->setProperty("cssClass", "proppanel");
			pComboBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
			pComboBox->addItems(items);
			pComboBox->setItemDelegate(new ZComboBoxItemDelegate(pComboBox));
			pComboBox->setObjectName(paramName);
			pComboBox->setProperty("control", param.control);
			pHLayout->addWidget(pComboBox);

			connect(pComboBox, &QComboBox::textActivated, this, &ZenoPropPanel::onLineEditFinish);

			pLayout->addLayout(pHLayout);
			break;
		}
		case CONTROL_READPATH:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(paramName);
			pNameItem->setProperty("cssClass", "proppanel");
			pHLayout->addWidget(pNameItem);

			QLineEdit* pathLineEdit = new QLineEdit(param.value.toString());
			pathLineEdit->setProperty("cssClass", "proppanel");
			pathLineEdit->setObjectName(paramName);
			pathLineEdit->setProperty("control", param.control);
			pHLayout->addWidget(pathLineEdit);
			connect(pathLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onLineEditFinish);

			ZIconLabel* openBtn = new ZIconLabel;
			openBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(28, 28)), ":/icons/ic_openfile.svg", ":/icons/ic_openfile-on.svg", ":/icons/ic_openfile-on.svg");
			pHLayout->addWidget(openBtn);

			pLayout->addLayout(pHLayout);
			break;
		}
		case CONTROL_WRITEPATH:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(paramName);
			pNameItem->setProperty("cssClass", "proppanel");
			pHLayout->addWidget(pNameItem);

			QLineEdit* pathLineEdit = new QLineEdit(param.value.toString());
			pathLineEdit->setProperty("cssClass", "proppanel");
			pathLineEdit->setObjectName(paramName);
			pathLineEdit->setProperty("control", param.control);
			pHLayout->addWidget(pathLineEdit);
			connect(pathLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onLineEditFinish);

			ZIconLabel* openBtn = new ZIconLabel;
			openBtn->setIcons(ZenoStyle::dpiScaledSize(QSize(28, 28)), ":/icons/ic_openfile.svg", ":/icons/ic_openfile-on.svg", ":/icons/ic_openfile-on.svg");
			pHLayout->addWidget(openBtn);

			pLayout->addLayout(pHLayout);
			break;
		}
		case CONTROL_MULTILINE_STRING:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(paramName);
			pNameItem->setProperty("cssClass", "proppanel");
			pHLayout->addWidget(pNameItem);

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

			pHLayout->addWidget(pTextEdit);

			pLayout->addLayout(pHLayout);
			break;
		}
		case CONTROL_HEATMAP:
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel("color");
			pNameItem->setProperty("cssClass", "proppanel");
			pHLayout->addWidget(pNameItem);

			QPushButton* pBtn = new QPushButton("Edit");
			pBtn->setObjectName("grayButton");
			pHLayout->addWidget(pBtn);

			pLayout->addLayout(pHLayout);
			break;
		}
		default:
		{
			break;
		}
		}
	}

	pParamsBox->setLayout(pLayout);
	pParamsBox->setTitle("paramters");

	return pParamsBox;
}

QGroupBox* ZenoPropPanel::inputsBox(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes)
{
	INPUT_SOCKETS inputs = pModel->data2(subgIdx, nodes[0], ROLE_INPUTS).value<INPUT_SOCKETS>();
	if (inputs.keys().isEmpty())
		return nullptr;

	QVBoxLayout* pLayout = new QVBoxLayout;
	for (QString inputSock : inputs.keys())
	{
		Q_ASSERT(inputs.find(inputSock) != inputs.end());
		INPUT_SOCKET input = inputs[inputSock];
		const QString& sockType = input.info.type;
		if (sockType == "int" || sockType == "float" || sockType == "vec3f" || sockType == "string")
		{
			QHBoxLayout* pHLayout = new QHBoxLayout;

			QLabel* pNameItem = new QLabel(inputSock);
			pNameItem->setProperty("cssClass", "proppanel");

			pHLayout->addWidget(pNameItem);

			QLineEdit* pLineEdit = new QLineEdit(UiHelper::variantToString(input.info.defaultValue));
			pLineEdit->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
			pLineEdit->setProperty("cssClass", "proppanel");
			pLineEdit->setObjectName(inputSock);
			pLineEdit->setProperty("control", input.info.control);

			pHLayout->addWidget(pLineEdit);

			connect(pLineEdit, &QLineEdit::editingFinished, this, &ZenoPropPanel::onLineEditFinish);
			pLayout->addLayout(pHLayout);
		}
	}

	if (pLayout->count() == 0)
	{
		delete pLayout;
		return nullptr;
	}

	QGroupBox* pInputsBox = new QGroupBox;
	pInputsBox->setLayout(pLayout);
	pInputsBox->setTitle("inputs");
	return pInputsBox;
}

void ZenoPropPanel::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
}

void ZenoPropPanel::onLineEditFinish()
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
				case CONTROL_BOOL:
				case CONTROL_READPATH:
				case CONTROL_WRITEPATH:
				{
					//update lineedit
					auto lst = findChildren<QLineEdit*>(param.name, Qt::FindDirectChildrenOnly);
					if (lst.size() == 1)
					{
						QLineEdit* pEdit = lst[0];
						pEdit->setText(param.value.toString());
					}
					break;
				}
				case CONTROL_ENUM:
				{
					auto lst = findChildren<QComboBox*>(param.name, Qt::FindDirectChildrenOnly);
					if (lst.size() == 1)
					{
						QComboBox* pCombo = lst[0];
						pCombo->setCurrentText(param.value.toString());
					}
					break;
				}
				case CONTROL_MULTILINE_STRING:
				{
					auto lst = findChildren<QTextEdit*>(param.name, Qt::FindDirectChildrenOnly);
					if (lst.size() == 1)
					{
						QTextEdit* pTextEdit = lst[0];
						pTextEdit->setText(param.value.toString());
					}
					break;
				}
			}
		}
	}
}