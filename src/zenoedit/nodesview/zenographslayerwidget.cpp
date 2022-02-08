#include "zenographslayerwidget.h"
#include <comctrl/ziconbutton.h>
#include <zenoui/model/graphsmodel.h>
#include <zenoui/model/modelrole.h>
#include "../zenoapplication.h"
#include "../graphsmanagment.h"
#include "../nodesys/zenosubgraphview.h"


LayerPathWidget::LayerPathWidget(QWidget* parent)
	: QWidget(parent)
{
	QHBoxLayout* pLayout = new QHBoxLayout;
	setLayout(pLayout);
}

void LayerPathWidget::setPath(const QString& path)
{
	m_path = path;
	QHBoxLayout* pLayout = qobject_cast<QHBoxLayout*>(this->layout());
	while (QWidget* w = findChild<QWidget*>())
	{
		delete w;
	}
	int cnt = pLayout->count();

	QStringList L = m_path.split("/");
	for (QString item : L)
	{
		if (item.isEmpty())
			continue;
		QColor clrHovered, clrSelected;
		clrHovered = QColor(67, 67, 67);
		clrSelected = QColor(33, 33 ,33);
		ZIconButton* pLabel = new ZIconButton(QIcon(), QSize(), clrHovered, clrSelected);
		pLabel->setText(item);
		QPalette pal = pLabel->palette();
		pal.setColor(pLabel->foregroundRole(), QColor(255,255,255));
		pLabel->setPalette(pal);
		ZIconButton* pArrow = new ZIconButton(QIcon(":/icons/dir_arrow.svg"), QSize(16, 28), clrHovered, clrSelected);
		pLayout->addWidget(pLabel);
		if (L.indexOf(item) != L.length() - 1)
			pLayout->addWidget(pArrow);
	}
	update();
}


////////////////////////////////////////////////////////////////////
ZenoStackedViewWidget::ZenoStackedViewWidget(QWidget* parent)
	: QStackedWidget(parent)
{
}

ZenoStackedViewWidget::~ZenoStackedViewWidget()
{
}

void ZenoStackedViewWidget::activate(const QString& subGraph, const QString& nodeId)
{
	auto graphsMgm = zenoApp->graphsManagment();
	if (m_views.find(subGraph) == m_views.end())
	{
		ZenoSubGraphScene* pScene = graphsMgm->scene(subGraph);
		ZenoSubGraphView* pView = new ZenoSubGraphView;
		pView->initScene(pScene);
		m_views[subGraph] = pView;
		addWidget(pView);
	}
	setCurrentWidget(m_views[subGraph]);
	SubGraphModel* pModel = graphsMgm->currentModel()->subGraph(subGraph);
	const QModelIndex& idx = pModel->index(nodeId);
	if (idx.isValid())
	{
		QPointF pos = pModel->data(idx, ROLE_OBJPOS).toPointF();
		m_views[subGraph]->focusOn(nodeId, pos);
	}
}


///////////////////////////////////////////////////////////////////////
ZenoGraphsLayerWidget::ZenoGraphsLayerWidget(QWidget* parent)
	: QWidget(parent)
	, m_pPathWidget(nullptr)
	, m_graphsWidget(nullptr)
{
	ZIconButton* pForward = new ZIconButton(QIcon(":/icons/forward.svg"), QSize(28, 28), QColor(), QColor());
	ZIconButton* pBackward = new ZIconButton(QIcon(":/icons/backward.svg"), QSize(28, 28), QColor(), QColor());
	m_pPathWidget = new LayerPathWidget;
	m_pPathWidget->hide();
	QHBoxLayout* pHLayout = new QHBoxLayout;
	pHLayout->addWidget(pForward);
	pHLayout->addWidget(pBackward);
	pHLayout->addWidget(m_pPathWidget);
	QVBoxLayout* pLayout = new QVBoxLayout;
	pLayout->addLayout(pHLayout);
	m_graphsWidget = new ZenoStackedViewWidget;
	pLayout->addWidget(m_graphsWidget);
	setLayout(pLayout);
}

void ZenoGraphsLayerWidget::resetPath(const QString& path, const QString& nodeId)
{
	m_pPathWidget->setPath(path);
	m_pPathWidget->show();
	QStringList L = path.split("/", Qt::SkipEmptyParts);
	const QString& subGraph = L[L.length() - 1];
	m_graphsWidget->activate(subGraph, nodeId);
}