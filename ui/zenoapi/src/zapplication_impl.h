#ifndef __ZAPPLICATION_IMPL_H__
#define __ZAPPLICATION_IMPL_H__

#include "../include/interface.h"

class IGraphsModel;

class ZApplication_Impl : public IZApplication
{
public:
    ZApplication_Impl();

    void clear() override;
    void openFile(const std::string& filePath) override;

    int count() const override;
    std::shared_ptr<IZSubgraph> item(int idx) override;

    std::shared_ptr<IZSubgraph> getSubgraph(const std::string& name) override;
    std::shared_ptr<IZSubgraph> addSubgraph(const std::string& name) override;
    bool removeSubgraph(const std::string& name) override;
    std::string forkSubgraph(const std::string& name) override;

private:
    IGraphsModel* m_model;
};

#endif