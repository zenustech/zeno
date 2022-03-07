#include "zenographslayerwidget.h"
#include <comctrl/ziconbutton.h>
#include "model/graphsmodel.h"
#include <zenoui/model/modelrole.h>
#include "../zenoapplication.h"
#include "../graphsmanagment.h"
#include "../nodesys/zenosubgraphview.h"

// jiao wo cihou da shi
#ifdef QT_VERSION
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
#define QtSkipEmptyParts Qt::SkipEmptyParts
#define msc_unordered_map std::map
#else
#define QtSkipEmptyParts QString::SkipEmptyParts
#endif
#endif


LayerPathWidget::LayerPathWidget(QWidget* parent)
	: QWidget(parent)
	, m_pForward(nullptr)
	, m_pBackward(nullptr)
{
	m_pLayerWidget = qobject_cast<ZenoGraphsLayerWidget*>(parent);
	QHBoxLayout* pLayout = new QHBoxLayout;
	m_pForward = new ZIconButton(QIcon(":/icons/forward.svg"), QSize(28, 28), QColor(), QColor());
	m_pBackward = new ZIconButton(QIcon(":/icons/backward.svg"), QSize(28, 28), QColor(), QColor());
	pLayout->addWidget(m_pForward);
	pLayout->addWidget(m_pBackward);
	setLayout(pLayout);
}

void LayerPathWidget::setPath(const QString& path)
{
	m_path = path;
	QHBoxLayout* pLayout = qobject_cast<QHBoxLayout*>(this->layout());
	while (pLayout->count() > 2)
	{
		QLayoutItem* pItem = pLayout->itemAt(pLayout->count() - 1);
		QWidget* w = pItem->widget();
		if (w != m_pForward && w != m_pBackward)
		{
			pLayout->removeItem(pItem);
			delete w;
		}
	}

	QStringList L = m_path.split("/", Qt::SkipEmptyParts);
	for (int i = 0; i < L.length(); i++)
	{
		const QString& item = L[i];
		Q_ASSERT(!item.isEmpty());
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

		connect(pLabel, SIGNAL(clicked()), this, SLOT(onPathItemClicked()));
	}
	pLayout->addStretch();
	update();
}

QString LayerPathWidget::path() const
{
	return m_path;
}

void LayerPathWidget::onPathItemClicked()
{
	ZIconButton* pClicked = qobject_cast<ZIconButton*>(sender());
	QString path;
	QHBoxLayout* pLayout = qobject_cast<QHBoxLayout*>(this->layout());

	bool bStartDeleted = false;
	for (int i = 0; i < pLayout->count();)
	{
		QLayoutItem* pItem = pLayout->itemAt(i);
		QWidget* w = pItem->widget();
		if (bStartDeleted)
		{
			if (qobject_cast<ZIconButton*>(w))
			{
				pLayout->removeItem(pItem);
				delete w;
				continue;
			}
		}
		else
		{
			if (ZIconButton* pPathItem = qobject_cast<ZIconButton*>(w))
			{
				if (pPathItem != m_pForward && pPathItem != m_pBackward)
					path += "/" + pPathItem->text();
			}
		}
		if (w == pClicked)
		{
			bStartDeleted = true;
		}
		i++;
	}
	m_path = path;
	m_pLayerWidget->activeByPath(path);
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
	IGraphsModel* pGraphsModel = zenoApp->graphsManagment()->currentModel();
	Q_ASSERT(pGraphsModel);
	const QModelIndex& subgIdx = pGraphsModel->index(subGraph);

	if (m_views.find(subGraph) == m_views.end())
	{
		ZenoSubGraphScene* pScene = qobject_cast<ZenoSubGraphScene*>(pGraphsModel->scene(subgIdx));
		Q_ASSERT(pScene);
		ZenoSubGraphView* pView = new ZenoSubGraphView;
		pView->initScene(pScene);
		m_views[subGraph] = pView;
		addWidget(pView);
	}
	setCurrentWidget(m_views[subGraph]);

	const QModelIndex& idx = pGraphsModel->index(nodeId, subgIdx);
	if (idx.isValid())
	{
		QPointF pos = pGraphsModel->data2(subgIdx, idx, ROLE_OBJPOS).toPointF();
		m_views[subGraph]->focusOn(nodeId, pos);
	}
}

void ZenoStackedViewWidget::clear()
{
	for (ZenoSubGraphView* view : m_views)
	{
		removeWidget(view);
	}
	m_views.clear();
}


///////////////////////////////////////////////////////////////////////
ZenoGraphsLayerWidget::ZenoGraphsLayerWidget(QWidget* parent)
	: QWidget(parent)
	, m_pPathWidget(nullptr)
	, m_graphsWidget(nullptr)
{
	QVBoxLayout* pLayout = new QVBoxLayout;
	m_pPathWidget = new LayerPathWidget(this);
	m_pPathWidget->hide();
	pLayout->addWidget(m_pPathWidget);
	m_graphsWidget = new ZenoStackedViewWidget;
	pLayout->addWidget(m_graphsWidget);
	pLayout->setMargin(0);
	setLayout(pLayout);
}

void ZenoGraphsLayerWidget::resetPath(const QString& path, const QString& nodeId)
{
	m_pPathWidget->setPath(path);
	m_pPathWidget->show();
	QStringList L = path.split("/", QtSkipEmptyParts);
	const QString& subGraph = L[L.length() - 1];
	m_graphsWidget->activate(subGraph, nodeId);
}

void ZenoGraphsLayerWidget::activeByPath(const QString& path)
{
	QStringList L = path.split("/", QtSkipEmptyParts);
	const QString& subGraph = L[L.length() - 1];
	m_graphsWidget->activate(subGraph, "");
}

void ZenoGraphsLayerWidget::clear()
{
	m_pPathWidget->hide();
	m_graphsWidget->clear();
}

QString ZenoGraphsLayerWidget::path() const
{
	return m_pPathWidget->path();
}
