#include "zenonewmenu.h"
#include "model/subgraphmodel.h"
#include "../graphsmanagment.h"
#include "zenoapplication.h"
#include "nodesmgr.h"
#include <zenoui/comctrl/gv/zenoparamwidget.h>


ZenoNewnodeMenu::ZenoNewnodeMenu(const QModelIndex& subgIdx, const NODE_CATES& cates, const QPointF& scenePos, QWidget* parent)
	: QMenu(parent)
	, m_cates(cates)
	, m_subgIdx(subgIdx)
	, m_scenePos(scenePos)
	, m_searchEdit(nullptr)
	, m_pWAction(nullptr)
{
	QVBoxLayout* pLayout = new QVBoxLayout;

	m_pWAction = new QWidgetAction(this);
	m_searchEdit = new ZenoGvLineEdit;
	m_searchEdit->setAutoFillBackground(false);
	m_searchEdit->setTextMargins(QMargins(8, 0, 0, 0));

	QPalette palette;
	palette.setColor(QPalette::Base, QColor(37, 37, 37));
	QColor clr = QColor(255, 255, 255);
	palette.setColor(QPalette::Text, clr);

	m_searchEdit->setPalette(palette);
	m_searchEdit->setFont(QFont("HarmonyOS Sans SC", 10));
	m_pWAction->setDefaultWidget(m_searchEdit);
	addAction(m_pWAction);

	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	QList<QAction*> actions = NodesMgr::getCategoryActions(pModel, m_subgIdx, "", m_scenePos);
	addActions(actions);

	connect(m_searchEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onTextChanged(const QString&)));
}

ZenoNewnodeMenu::~ZenoNewnodeMenu()
{
}

void ZenoNewnodeMenu::setEditorFocus()
{
	m_searchEdit->setFocus();
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
	setEditorFocus();
}
