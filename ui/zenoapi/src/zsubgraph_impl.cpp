#include "zsubgraph_impl.h"
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>


ZSubGraph_Impl::ZSubGraph_Impl(IGraphsModel* pModel, const QModelIndex& idx)
    : m_index(idx)
    , m_model(pModel)
{
}

ZSubGraph_Impl::~ZSubGraph_Impl()
{

}

std::string ZSubGraph_Impl::name() const
{
    if (m_index.isValid())
        return m_index.data(ROLE_OBJNAME).toString().toStdString();
    else
        return "";
}

std::shared_ptr<IZNode> ZSubGraph_Impl::getNode(const std::string& ident)
{
    if (!m_index.isValid() || !m_model)
        return nullptr;
    
    QModelIndex idx = m_model->index(QString::fromStdString(ident), m_index);
    if (!idx.isValid())
        return nullptr;



}

std::shared_ptr<IZNode> ZSubGraph_Impl::addNode(const std::string& nodeCls)
{
    return nullptr;
}

bool ZSubGraph_Impl::addLink(
    const std::string& outIdent,
    const std::string& outSock,
    const std::string& inIdent,
    const std::string& inSock)
{
    return false;
}

bool ZSubGraph_Impl::removeLink(
    const std::string& outIdent,
    const std::string& outSock,
    const std::string& inIdent,
    const std::string& inSock)
{
    return false;
}

int ZSubGraph_Impl::count() const
{
    return 0;
}

std::shared_ptr<IZNode> ZSubGraph_Impl::item(int idx)
{
    return nullptr;
}