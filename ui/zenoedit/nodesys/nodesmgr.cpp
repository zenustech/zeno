#include "nodesmgr.h"
#include <zenoui/util/uihelper.h>
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

	pModel->beginTransaction("add node");
	pModel->addNode(node, subgIdx, true);
	pModel->endTransaction();
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

bool key_appear_by_order(const QString& pattern, QString key) {
	for (auto i = 0; i < pattern.size(); i++) {
		QChar c = pattern.at(i);
		auto res = key.indexOf(c, 0, Qt::CaseInsensitive);
		if (res == -1) {
			return false;
		}
		key = key.mid(res + 1);
	}
	return true;
}

bool capital_match(const QString& pattern, const QString& key) {
	QString only_upper;
	for (auto i = 0; i < key.size(); i++) {
		if (key.at(i).isUpper()) {
			only_upper += key.at(i);
		}
	}
	return only_upper.contains(pattern, Qt::CaseInsensitive);
}

void merge_condidates(
	QList<QString>& ret_list,
	QSet<QString>& ret_set,
	const QList<QString>& lst
) {
	const int MAX_COUNT = 30;
	for (auto i = 0; i < lst.size(); i++) {
		if (ret_list.size() > MAX_COUNT) {
			break;
		}
		auto s = lst[i];
		if (!ret_set.contains(s)) {
			ret_list.push_back(s);
			ret_set.insert(s);
		}
	}
}

QList<QString> fuzzy_search(const QString& pattern, const QList<QString>& keys) {
	QList<QString> key_appear_by_order_conds;
	for (auto i = 0; i < keys.size(); i++) {
		auto k = keys[i];
		if (key_appear_by_order(pattern, k)) {
			key_appear_by_order_conds.push_back(k);
		}
	}
	QList<QString> direct_match_conds;
	for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
		auto k = key_appear_by_order_conds[i];
		if (k.contains(pattern, Qt::CaseInsensitive)) {
			direct_match_conds.push_back(k);
		}
	}
	QList<QString> prefix_match_conds;
	for (auto i = 0; i < direct_match_conds.size(); i++) {
		auto k = direct_match_conds[i];
		if (k.indexOf(pattern, 0, Qt::CaseInsensitive) == 0) {
			prefix_match_conds.push_back(k);
		}
	}
	QList<QString> capital_match_conds;
	for (auto i = 0; i < key_appear_by_order_conds.size(); i++) {
		auto k = key_appear_by_order_conds[i];
		if (capital_match(pattern, k)) {
			capital_match_conds.push_back(k);
		}
	}
	QList<QString> ret_list;
	QSet<QString> ret_set;
	merge_condidates(ret_list, ret_set, prefix_match_conds);
	merge_condidates(ret_list, ret_set, capital_match_conds);
	merge_condidates(ret_list, ret_set, direct_match_conds);
	merge_condidates(ret_list, ret_set, key_appear_by_order_conds);
	return ret_list;
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