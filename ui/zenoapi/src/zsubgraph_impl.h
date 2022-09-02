#ifndef __ZSUBGRAPH_IMPL_H__
#define __ZSUBGRAPH_IMPL_H__

#include "../include/interface.h"
#include <QAbstractItemModel>

class IGraphsModel;

class ZSubGraph_Impl : public IZSubgraph
{
public:
    ZSubGraph_Impl(IGraphsModel* pModel, const QModelIndex& idx);
    ~ZSubGraph_Impl();

    std::string name() const override;
    std::shared_ptr<IZNode> getNode(const std::string& ident) override;
    std::shared_ptr<IZNode> addNode(const std::string& nodeCls) override;

    bool addLink(
        const std::string& outIdent,
        const std::string& outSock,
        const std::string& inIdent,
        const std::string& inSock) override;

    bool removeLink(
        const std::string& outIdent,
        const std::string& outSock,
        const std::string& inIdent,
        const std::string& inSock) override;

    int count() const override;
    std::shared_ptr<IZNode> item(int idx) override;

private:
    QPersistentModelIndex m_index;
    IGraphsModel* m_model;
};

#endif