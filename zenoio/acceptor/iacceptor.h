#ifndef __IACCEPTOR_H__
#define __IACCEPTOR_H__

#include <model/modeldata.h>

interface IAcceptor
{
    virtual void setDescriptors(const NODE_DESCS& nodesParams) = 0;
    virtual void BeginSubgraph(const QString& name) = 0;
    virtual void EndSubgraph() = 0;
    virtual void switchSubGraph(const QString& graphName) = 0;
    virtual void addNodeData()
};


#endif