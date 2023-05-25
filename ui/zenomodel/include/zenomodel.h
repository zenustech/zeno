#ifndef __ZENO_MODEL_H__
#define __ZENO_MODEL_H__

#include <zenomodel/include/igraphsmodel.h>
#include <zenoio/acceptor/iacceptor.h>

namespace zeno_model
{
    IGraphsModel* createModel(bool bSharedModel, QObject* parent);
    IAcceptor* createIOAcceptor(IGraphsModel* pNodeModel, IGraphsModel* pSubgraphModel, bool bImport);
}

#endif