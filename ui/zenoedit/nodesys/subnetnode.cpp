#include "subnetnode.h"
#include "model/graphsmodel.h"
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/util/uihelper.h>


SubnetNode::SubnetNode(bool bInput, const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent)
	, m_bInput(bInput)
{

}

SubnetNode::~SubnetNode()
{

}

void SubnetNode::onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue)
{
	//get old name first.
	IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
	Q_ASSERT(pModel);
	const QString& nodeid = nodeId();
	QModelIndex subgIdx = this->subGraphIndex();
	const PARAMS_INFO& params = pModel->data2(subgIdx, index(), ROLE_PARAMETERS).value<PARAMS_INFO>();
	const QString& oldName = params["name"].value.toString();
	const QString& subnetName = pModel->name(subgIdx);
	if (oldName == textValue)
		return;

	ZenoNode::onParamEditFinished(editCtrl, paramName, textValue);
}