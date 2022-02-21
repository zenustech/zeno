#include "subnetnode.h"
#include <model/graphsmodel.h>
#include "zenoapplication.h"
#include "graphsmanagment.h"
#include <zenoui/util/uihelper.h>


SubInputNode::SubInputNode(const NodeUtilParam& params, QGraphicsItem* parent)
	: ZenoNode(params, parent)
{

}

SubInputNode::~SubInputNode()
{

}

void SubInputNode::onParamEditFinished(PARAM_CONTROL editCtrl, const QString& paramName, const QString& textValue)
{
    //get old name first.
    IGraphsModel* pModel = zenoApp->graphsManagment()->currentModel();
    Q_ASSERT(pModel);
	const QString& nodeid = nodeId();
	QModelIndex subgIdx = this->subGraphIndex();
    const PARAMS_INFO& subInputs = pModel->data2(subgIdx, index(), ROLE_PARAMETERS).value<PARAMS_INFO>();
    const QString& oldName = subInputs["name"].value.toString();
	const QString& name = pModel->name(subgIdx);
	if (paramName != "name" || oldName == textValue)
		return;

	QVariant newValue = UiHelper::parseTextValue(editCtrl, textValue);

	pModel->beginTransaction("update socket");

	PARAM_UPDATE_INFO info;
	info.oldValue = pModel->getParamValue(nodeid, paramName, subgIdx);
	info.newValue = newValue;
	info.name = paramName;
	pModel->updateParamInfo(nodeid, info, subgIdx, true);

	for (int r = 0; r < pModel->rowCount(); r++)
	{
		subgIdx = pModel->index(r, 0);
		QModelIndexList m_results = pModel->searchInSubgraph(name, subgIdx);
        for (auto idx : m_results)
        {
            SOCKET_INFO sock;
			sock.name = textValue;
			//todo: type 

			SOCKET_UPDATE_INFO info;
			info.bInput = true;
			info.oldInfo.name = oldName;
			info.name = textValue;
			info.newInfo = sock;
			pModel->updateSocket(idx.data(ROLE_OBJID).toString(), info, subgIdx, true);
        }
	}

	pModel->endTransaction();
}