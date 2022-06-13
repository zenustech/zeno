#include "apphelper.h"
#include <zenoui/model/modeldata.h>
#include <zenoui/model/modelrole.h>
#include "util/log.h"


QString AppHelper::correctSubIOName(IGraphsModel* pModel, const QString& subgName, const QString& newName, bool bInput)
{
    ZASSERT_EXIT(pModel, "");

    NODE_DESCS descs = pModel->descriptors();
    if (descs.find(subgName) == descs.end())
        return "";

    const NODE_DESC &desc = descs[subgName];
    QString finalName = newName;
    int i = 1;
    if (bInput)
    {
        while (desc.inputs.find(finalName) != desc.inputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    else
    {
        while (desc.outputs.find(finalName) != desc.outputs.end())
        {
            finalName = finalName + QString("_%1").arg(i);
            i++;
        }
    }
    return finalName;
}


QModelIndex AppHelper::getSubInOutNode(IGraphsModel* pModel, const QModelIndex& subgIdx, const QString& subName, bool bInput)
{
    const QList<QModelIndex>& indices = pModel->searchInSubgraph(bInput ? "SubInput" : "SubOutput", subgIdx);
    for (const QModelIndex &idx_ : indices)
    {
        const QString &subInputId = idx_.data(ROLE_OBJID).toString();
        const PARAMS_INFO &params = idx_.data(ROLE_PARAMETERS).value<PARAMS_INFO>();
        if (params["name"].value == subName)
        {
            return idx_;
        }
    }
    return QModelIndex();
}