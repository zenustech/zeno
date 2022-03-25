#include "zenoproppanel.h"
#include <zenoui/model/modelrole.h>
#include <zenoui/include/igraphsmodel.h>


ZenoPropPanel::ZenoPropPanel(QWidget* parent)
    : QWidget(parent)
{
}

ZenoPropPanel::~ZenoPropPanel()
{
}

void ZenoPropPanel::init(IGraphsModel* pModel, const QModelIndex& subgIdx, const QModelIndexList& nodes, bool select)
{
    setUpdatesEnabled(false);
	qDeleteAll(findChildren<QWidget*>("", Qt::FindDirectChildrenOnly));
	setUpdatesEnabled(true);

    if (!pModel || !select)
        return;

    QVBoxLayout* pLayout = new QVBoxLayout;
    
    PARAMS_INFO params = pModel->data2(subgIdx, nodes[0], ROLE_PARAMETERS).value<PARAMS_INFO>();
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
				pHLayout->addWidget(pNameItem);

				QLineEdit* pLineEdit = new QLineEdit(param.value.toString());
				pHLayout->addWidget(pLineEdit);

				connect(pLineEdit, &QLineEdit::editingFinished, this, [=]() {
					int j;
					j = 0;
				});
				pLayout->addLayout(pHLayout);
				break;
			}
			case CONTROL_ENUM:
			{
				QHBoxLayout* pHLayout = new QHBoxLayout;

				QLabel* pNameItem = new QLabel(paramName);
				pHLayout->addWidget(pNameItem);

				QStringList items = param.typeDesc.mid(QString("enum ").length()).split(QRegExp("\\s+"));
				QComboBox* pComboBox = new QComboBox;
				pComboBox->addItems(items);
				pHLayout->addWidget(pComboBox);

				connect(pComboBox, &QComboBox::textActivated, this, [=](const QString& textValue) {
					int j;
					j = 0;
				});
				
				pLayout->addLayout(pHLayout);
				break;
			}
			case CONTROL_READPATH:
			{
				QHBoxLayout* pHLayout = new QHBoxLayout;

				QLabel* pNameItem = new QLabel(paramName);
				pHLayout->addWidget(pNameItem);

				QLineEdit* pFileWidget = new QLineEdit(param.value.toString());
				pHLayout->addWidget(pFileWidget);

				//ImageElement elem;
				//elem.image = ":/icons/ic_openfile.svg";
				//elem.imageHovered = ":/icons/ic_openfile-on.svg";
				//elem.imageOn = ":/icons/ic_openfile-on.svg";
				QPushButton* openBtn = new QPushButton("...");
				pHLayout->addWidget(openBtn);

				break;
			}
			case CONTROL_WRITEPATH:
			{
				QHBoxLayout* pHLayout = new QHBoxLayout;

				QLabel* pNameItem = new QLabel(paramName);
				pHLayout->addWidget(pNameItem);

				QLineEdit* pFileWidget = new QLineEdit(param.value.toString());
				pHLayout->addWidget(pFileWidget);

				//ImageElement elem;
				//elem.image = ":/icons/ic_openfile.svg";
				//elem.imageHovered = ":/icons/ic_openfile-on.svg";
				//elem.imageOn = ":/icons/ic_openfile-on.svg";
				QPushButton* openBtn = new QPushButton("...");
				pHLayout->addWidget(openBtn);
				break;
			}
			case CONTROL_MULTILINE_STRING:
			{
				break;
			}
			case CONTROL_HEAPMAP:
			{
				//break;
			}
			default:
			{
				break;
			}
        }
    }

	setLayout(pLayout);
}

void ZenoPropPanel::mousePressEvent(QMouseEvent* event)
{
	QWidget::mousePressEvent(event);
}