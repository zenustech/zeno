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
	, m_pNewSubnetBtn(nullptr)
	, m_bListView(true)
{
	QVBoxLayout* pMainLayout = new QVBoxLayout;

	m_pListView = new ZenoSubnetListView;
	m_pTreeView = new ZenoSubnetTreeView;
	pMainLayout->addWidget(m_pListView);
	pMainLayout->addWidget(m_pTreeView);

	m_pNewSubnetBtn = new ZTextLabel("Add New Subnet");
	QFont font = QFont("HarmonyOS Sans", 12);
	font.setPointSize(13);
	font.setBold(false);
	m_pNewSubnetBtn->setFont(font);
	m_pNewSubnetBtn->setTextColor(QColor(116, 116, 116));
	m_pNewSubnetBtn->setBackgroundColor(QColor(56, 57, 56));
	m_pNewSubnetBtn->setAlignment(Qt::AlignCenter);
	m_pNewSubnetBtn->setFixedHeight(ZenoStyle::dpiScaled(40));

	pMainLayout->addWidget(m_pNewSubnetBtn);
	pMainLayout->setContentsMargins(0, 0, 0, 0);

	setLayout(pMainLayout);

	connect(m_pListView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
	connect(m_pTreeView, SIGNAL(clicked(const QModelIndex&)), this, SIGNAL(clicked(const QModelIndex&)));
	connect(m_pListView, SIGNAL(graphToBeActivated(const QString&)), this, SIGNAL(graphToBeActivated(const QString&)));
	connect(m_pNewSubnetBtn, SIGNAL(clicked()), this, SLOT(onNewSubnetBtnClicked()));
	setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);

	m_pListView->setVisible(m_bListView);
	m_pTreeView->setVisible(!m_bListView);
}

void ZenoSubnetPanel::initModel(IGraphsModel* pModel)
{
	GraphsTreeModel* pTreeModel = zenoApp->graphsManagment()->treeModel();
	m_pListView->initModel(pModel);
	m_pTreeView->initModel(pTreeModel);
	QString fn = pModel->fileName();
	if (fn.isEmpty())
		fn = "newFile";
	connect(pModel, SIGNAL(modelReset()), this, SLOT(onModelReset()));
}

void ZenoSubnetPanel::setViewWay(bool bListView)
{
	m_bListView = bListView;
	m_pListView->setVisible(m_bListView);
	m_pTreeView->setVisible(!m_bListView);
	m_pNewSubnetBtn->setVisible(m_bListView);
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
