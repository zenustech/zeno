#ifndef __ZENO_MODEL_H__
#define __ZENO_MODEL_H__

#include <zenomodel/include/igraphsmodel.h>

namespace zeno_model
{
    IGraphsModel* createModel(QObject* parent);
    QAbstractItemModel* treeModel(IGraphsModel* pModel, QObject* parent);
}

#endif