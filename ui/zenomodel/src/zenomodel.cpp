#include "../include/zenomodel.h"
#include "graphsmodel.h"
#include "graphstreemodel.h"
#include "modelacceptor.h"

namespace zeno_model
{
    IGraphsModel* createModel(QObject* parent)
    {
        return new GraphsModel(parent);
    }

    IAcceptor* createIOAcceptor(IGraphsModel* pModel, bool bImport)
    {
        if (GraphsModel* model = qobject_cast<GraphsModel*>(pModel))
            return new ModelAcceptor(model, bImport);
        else
            return nullptr;
    }

    QAbstractItemModel* treeModel(IGraphsModel* pModel, QObject* parent)
    {
        GraphsTreeModel* pTreeModel = new GraphsTreeModel(parent);
        pTreeModel->init(pModel);
        return pTreeModel;
    }
}