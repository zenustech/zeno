#include "zapplication_impl.h"
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/zenomodel.h>
#include <zenoio/reader/zsgreader.h>
#include <zassert.h>


class IOBreakingScope
{
public:
    IOBreakingScope(IGraphsModel* model) : m_model(model) {
        if (m_model)
            m_model->setIOProcessing(true);
    }

    ~IOBreakingScope() {
        if (m_model)
            m_model->setIOProcessing(false);
    }

private:
    IGraphsModel* m_model;
};


ZApplication_Impl::ZApplication_Impl()
    : m_model(nullptr)
{
}

void ZApplication_Impl::clear()
{

}

void ZApplication_Impl::openFile(const std::string& filePath)
{
    if (m_model)
        m_model->clear();

    //todo: parent.
    m_model = zeno_model::createModel(nullptr);
    IOBreakingScope batch(m_model);
    std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(m_model, false));
    if (!ZsgReader::getInstance().openFile(QString::fromStdString(filePath), acceptor.get()))
        return;

    m_model->clearDirty();
}

int ZApplication_Impl::count() const
{
    return m_model->rowCount();
}

std::shared_ptr<IZSubgraph> ZApplication_Impl::item(int idx)
{
    if (idx < 0 || idx >= count())
        return nullptr;

    QModelIndex& subgIdx = m_model->index(idx, 0);

    return nullptr;
}

std::shared_ptr<IZSubgraph> ZApplication_Impl::getSubgraph(const std::string& name)
{
    return nullptr;
}

std::shared_ptr<IZSubgraph> ZApplication_Impl::addSubgraph(const std::string& name)
{
    return nullptr;
}

bool ZApplication_Impl::removeSubgraph(const std::string& name)
{
    return false;
}

std::string ZApplication_Impl::forkSubgraph(const std::string& name)
{
    return "";
}
