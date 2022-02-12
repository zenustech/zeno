#include "subnetnode.h"
#include <model/graphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"


SubInputNode::SubInputNode(const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent)
{

}

SubInputNode::~SubInputNode()
{

}

void SubInputNode::onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue)
{
	_base::onParamEditFinished(editCtrl, paramName, textValue);

	/*
	QAbstractItemModel* pSubModel_ = const_cast<QAbstractItemModel*>(index().model());
	SubGraphModel* pSubModel = qobject_cast<SubGraphModel*>(pSubModel_);
	const QString& name = pSubModel->name();

	//get old name first.
	const PARAMS_INFO& subInputs = pSubModel->data(pSubModel->index(nodeId()), ROLE_PARAMETERS).value<PARAMS_INFO>();
	const QString& oldName = subInputs["name"].value.toString();

	if (paramName != "name")
		return;

	auto graphsGM = zenoApp->graphsManagment();
	GraphsModel* pModel = graphsGM->currentModel();

	for (int r = 0; r < pModel->rowCount(); r++)
	{
		QModelIndex index = pModel->index(r, 0);
		Q_ASSERT(index.isValid());
		SubGraphModel* pSubModel = static_cast<SubGraphModel*>(index.data(ROLE_GRAPHPTR).value<void*>());

		QModelIndexList m_results = pSubModel->match(pSubModel->index(0, 0), ROLE_OBJNAME, name, -1, Qt::MatchContains);
		for (auto idx : m_results)
		{
			SOCKET_INFO info;
			info.name = textValue;
			pSubModel->updateSocket(idx.data(ROLE_OBJID).toString(), oldName, info);
		}
	}
	*/
}