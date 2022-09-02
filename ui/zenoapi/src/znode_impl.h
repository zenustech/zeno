#ifndef __ZNODE_IMPL_H__
#define __ZNODE_IMPL_H__

#include "interface.h"
#include <QAbstractItemModel>

class IGraphsModel;

class ZNode_Impl : public IZNode
{
public:
    ZNode_Impl(IGraphsModel* pModel, const QModelIndex& idx);
    ~ZNode_Impl();

    std::string getName() const override;
    std::string getIdent() const override;
    ZVARIANT getSocketDefl(const std::string& sockName) override;
    void setSocketDefl(const std::string& sockName, const ZVARIANT& value) override;
    ZVARIANT getParam(const std::string& name) override;
    void setParamValue(const std::string& name, const ZVARIANT& value) override;

private:
    QPersistentModelIndex m_index;
    IGraphsModel* m_model;
};


#endif