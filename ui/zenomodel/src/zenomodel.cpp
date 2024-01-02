#include "../include/zenomodel.h"
#include "graphsmodel.h"
#include "graphstreemodel.h"

namespace zeno_model
{
    IGraphsModel* createModel(QObject* parent)
    {
        return new GraphsModel(parent);
    }

    QAbstractItemModel* treeModel(IGraphsModel* pModel, QObject* parent)
    {
        GraphsTreeModel* pTreeModel = new GraphsTreeModel(parent);
        pTreeModel->init(pModel);
        return pTreeModel;
    }
}