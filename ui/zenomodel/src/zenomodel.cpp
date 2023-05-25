#include "../include/zenomodel.h"
#include "graphsmodel.h"
#include "graphstreemodel.h"
#include "modelacceptor.h"
#include "treeacceptor.h"

namespace zeno_model
{
    IGraphsModel* createModel(bool bSharedModel, QObject* parent)
    {
        if (bSharedModel)
            return new GraphsModel(parent);
        else
            return new GraphsTreeModel(parent);
    }

    IAcceptor* createIOAcceptor(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphModel, bool bImport)
    {
        if (GraphsModel *model = qobject_cast<GraphsModel *>(pNodeModel))
        {
            return new ModelAcceptor(model, bImport);
        }
        else if (GraphsTreeModel *model = qobject_cast<GraphsTreeModel *>(pNodeModel))
        {
            GraphsModel* pSubgraphs = qobject_cast<GraphsModel*>(pSubgraphModel);
            ZASSERT_EXIT(pSubgraphs, nullptr);
            return new TreeAcceptor(model, pSubgraphs, bImport);
        }
        else
            return nullptr;
    }
}