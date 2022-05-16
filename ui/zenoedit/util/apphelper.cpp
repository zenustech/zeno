#include "apphelper.h"
#include <zenoui/model/modeldata.h>
#include <zenoui/model/modelrole.h>
#include "util/log.h"


void AppHelper::correctSubIOName(IGraphsModel* pModel, QModelIndex subgIdx, const QString& descName, PARAMS_INFO& params)
{
    if (descName != "SubInput" && descName != "SubOutput")
        return;

    ZASSERT_EXIT(params.find("name") != params.end());
    QString name = params["name"].value.toString();

    QModelIndexList results = pModel->searchInSubgraph(descName, subgIdx);
    QStringList nameList;
    for (auto idx : results)
	{
        const PARAMS_INFO &_params = idx.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        ZASSERT_EXIT(_params.find("name") != _params.end());
        QString _name = _params["name"].value.toString();
        nameList.append(_name);
	}

	int i = 1;
	QString finalName = name;
	while (nameList.indexOf(finalName) != -1)
	{
		finalName = name + QString("(%1)").arg(i);
		i++;
	}
	params["name"].value = finalName;
}