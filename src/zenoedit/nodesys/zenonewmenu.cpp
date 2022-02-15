#include "zenonewmenu.h"
#include <zenoui/model/subgraphmodel.h>
#include "../graphsmanagment.h"
#include "zenoapplication.h"
#include "nodesmgr.h"


ZenoNewnodeMenu::ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, QWidget* parent)
	: QMenu(parent)
	, m_cates(cates)
	, m_subgIdx(subgIdx)
	, m_scenePos(scenePos)
	, m_searchEdit(nullptr)
	, m_pWAction(nullptr)
{
	QVBoxLayout* pLayout = new QVBoxLayout;
	m_searchEdit = new QLineEdit;

	m_pWAction = new QWidgetAction(this);
	QLineEdit* pSearchEdit = new QLineEdit;
	m_pWAction->setDefaultWidget(pSearchEdit);
	addAction(m_pWAction);

	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	QList<QAction*> actions = NodesMgr::getCategoryActions(pModel, m_subgIdx, "", m_scenePos);
	addActions(actions);

	connect(pSearchEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onTextChanged(const QString&)));
}

ZenoNewnodeMenu::~ZenoNewnodeMenu()
{
}

void ZenoNewnodeMenu::onTextChanged(const QString& text)
{
	QList<QAction*> acts = actions();

	for (int i = 0; i < acts.size(); i++) {
		if (acts[i] == m_pWAction) continue;
		removeAction(acts[i]);
		if (acts[i]->parent() == this)
			delete acts[i];
	}

	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	QList<QAction*> actions = NodesMgr::getCategoryActions(pModel, m_subgIdx, text, m_scenePos);
	addActions(actions);
}
