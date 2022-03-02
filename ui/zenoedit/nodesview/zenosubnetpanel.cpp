#include "zenosubnetpanel.h"
#include "zenosubnetlistview.h"
#include "zenoapplication.h"
#include "zenosubnettreeview.h"
#include <comctrl/zlabel.h>
#include "style/zenostyle.h"
#include "model/graphsmodel.h"
#include "graphsmanagment.h"
#include <zenoui/model/modelrole.h>


ZenoSubnetPanel::ZenoSubnetPanel(QWidget* parent)
	: QWidget(parent)
	, m_pListView(nullptr)
	, m_pTreeView(nullptr)
	, m_bListView(true)
{
	QVBoxLayout* pMainLayout = new QVBoxLayout;

	QHBoxLayout* pLabelLayout = new QHBoxLayout;
	QLabel* pIcon = new QLabel;
	pIcon->setPixmap(QIcon(":/icons/ic_File.svg").pixmap(ZenoStyle::dpiScaledSize(QSize(16, 16)), QIcon::Normal));

	m_pTextLbl = new QLabel("");
	QFont font = QFont("HarmonyOS Sans", 12);
	font.setBold(false);
	m_pTextLbl->setFont(font);
	QPalette pal = m_pTextLbl->palette();
	pal.setColor(QPalette::WindowText, QColor(128, 124, 122));
	m_pTextLbl->setPalette(pal);

	pLabelLayout->addWidget(pIcon);
	pLabelLayout->addWidget(m_pTextLbl);
	pLabelLayout->addStretch();
	pLabelLayout->setContentsMargins(12, 5, 5, 0);
	pMainLayout->addLayout(pLabelLayout);

	m_pListView = new ZenoSubnetListView;
	m_pTreeView = new ZenoSubnetTreeView;
	pMainLayout->addWidget(m_pListView);
	pMainLayout->addWidget(m_pTreeView);

	ZTextLabel* pNewSubnetBtn = new ZTextLabel("Add New Subnet");
	font.setPointSize(13);
	font.setBold(false);
	pNewSubnetBtn->setFont(font);
	pNewSubnetBtn->setTextColor(QColor(116, 116, 116));
	pNewSubnetBtn->setBackgroundColor(QColor(56, 57, 56));
	pNewSubnetBtn->setAlignment(Qt::AlignCenter);
	pNewSubnetBtn->setFixedHeight(ZenoStyle::dpiScaled(40));

	pMainLayout->addWidget(pNewSubnetBtn);
	pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(pMainLayout);

	connect(m_pListView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
	connect(m_pTreeView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
	connect(m_pListView, SIGNAL(graphToBeActivated(const QString&)), this, SIGNAL(graphToBeActivated(const QString&)));
	connect(pNewSubnetBtn, SIGNAL(clicked()), this, SLOT(onNewSubnetBtnClicked()));
	setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

	m_pListView->setVisible(m_bListView);
	m_pTreeView->setVisible(!m_bListView);
}

void ZenoSubnetPanel::initModel(IGraphsModel* pModel)
{
	GraphsTreeModel* pTreeModel = zenoApp->graphsManagment()->treeModel();
	m_pListView->initModel(pModel);
	m_pTreeView->initModel(pTreeModel);
	m_pTextLbl->setText(pModel->fileName());
	connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
}

void ZenoSubnetPanel::setViewWay(bool bListView)
{
	m_bListView = bListView;
	m_pListView->setVisible(m_bListView);
	m_pTreeView->setVisible(!m_bListView);
}

QSize ZenoSubnetPanel::sizeHint() const
{
	if (m_bListView)
	{
		int w = m_pListView->sizeHint().width();
		int h = QWidget::sizeHint().height();
		return QSize(w, h);
	}
	else
	{
		int w = m_pTreeView->sizeHint().width();
		int h = QWidget::sizeHint().height();
		return QSize(w, h);
	}
}

void ZenoSubnetPanel::onModelReset()
{
	hide();
}

void ZenoSubnetPanel::onNewSubnetBtnClicked()
{
	if (m_bListView)
	{
		m_pListView->edittingNew();
	}
}

void ZenoSubnetPanel::paintEvent(QPaintEvent* e)
{
	QPainter painter(this);
	painter.fillRect(rect(), QColor(42, 42, 42));
}
