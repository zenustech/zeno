#include "nodesmgr.h"
#include "fuzzy_search.h"
#include <zenoui/util/uihelper.h>
#include "util/apphelper.h"
#include <zeno/utils/log.h>


void NodesMgr::createNewNode(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, const QPointF& pt)
{
	zeno::log_debug("onNewNodeCreated");
	NODE_DESCS descs = pModel->descriptors();
	const NODE_DESC& desc = descs[descName];

	const QString& nodeid = UiHelper::generateUuid(descName);
	NODE_DATA node;
	node[ROLE_OBJID] = nodeid;
	node[ROLE_OBJNAME] = descName;
	node[ROLE_NODETYPE] = nodeType(descName);
	node[ROLE_INPUTS] = QVariant::fromValue(desc.inputs);
	node[ROLE_OUTPUTS] = QVariant::fromValue(desc.outputs);
	node[ROLE_PARAMETERS] = QVariant::fromValue(desc.params);
	node[ROLE_OBJPOS] = pt;
	node[ROLE_COLLASPED] = false;

	pModel->addNode(node, subgIdx, true);
}

NODE_TYPE NodesMgr::nodeType(const QString& name)
{
	if (name == "Blackboard")
	{
		return BLACKBOARD_NODE;
	}
	else if (name == "SubInput")
	{
		return SUBINPUT_NODE;
	}
	else if (name == "SubOutput")
	{
		return SUBOUTPUT_NODE;
	}
	else if (name == "MakeHeatmap")
	{
		return HEATMAP_NODE;
	}
	else
	{
		return NORMAL_NODE;
	}
}

QList<QAction*> NodesMgr::getCategoryActions(IGraphsModel* pModel, QModelIndex subgIdx, const QString& filter, QPointF scenePos)
{
	Q_ASSERT(pModel);
	if (!pModel)
		return QList<QAction*>();

	NODE_CATES cates = pModel->getCates();
	QList<QAction*> acts;
	if (cates.isEmpty())
	{
		QAction* pAction = new QAction("ERROR: no descriptors loaded!");
		pAction->setEnabled(false);
		acts.push_back(pAction);
		return acts;
	}

	if (!filter.isEmpty())
	{
		QList<QString> condidates;
		for (const NODE_CATE& cate : cates) {
			for (const QString& name : cate.nodes) {
				condidates.push_back(name);
			}
		}
		for(const QString& name: fuzzy_search(filter, condidates)) {
			QAction* pAction = new QAction(name);
			connect(pAction, &QAction::triggered, [=]() {
				createNewNode(pModel, subgIdx, name, scenePos);
			});
			acts.push_back(pAction);
		}
		return acts;
	}
	else
	{
		for (const NODE_CATE& cate : cates)
		{
			QAction* pAction = new QAction(cate.name);
			QMenu* pChildMenu = new QMenu;
			pChildMenu->setToolTipsVisible(true);
			for (const QString& name : cate.nodes)
			{
				QAction* pChildAction = pChildMenu->addAction(name);
				//todo: tooltip
				connect(pChildAction, &QAction::triggered, [=]() {
					createNewNode(pModel, subgIdx, name, scenePos);
				});
			}
			pAction->setMenu(pChildMenu);
			acts.push_back(pAction);
		}
	}
	return acts;
}
