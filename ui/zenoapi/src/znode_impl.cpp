#include "znode_impl.h"
#include <zenomodel/include/igraphsmodel.h>
#include <zenomodel/include/modeldata.h>
#include <zenomodel/include/modelrole.h>
#include "apiutil.h"


ZNode_Impl::ZNode_Impl(IGraphsModel* pModel, const QModelIndex& idx)
    : m_model(pModel)
    , m_index(idx)
{

}

ZNode_Impl::~ZNode_Impl()
{

}

std::string ZNode_Impl::getName() const
{
    if (!m_index.isValid())
        return "";
    return m_index.data(ROLE_OBJNAME).toString().toStdString();
}

std::string ZNode_Impl::getIdent() const
{
    if (!m_index.isValid())
        return "";
    return m_index.data(ROLE_OBJID).toString().toStdString();
}

ZVARIANT ZNode_Impl::getSocketDefl(const std::string& sockName)
{
    ZVARIANT var;

    QString qsName = QString::fromStdString(sockName);
    INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    if (inputs.find(qsName) == inputs.end())
        return var;

    INPUT_SOCKET socket = inputs[qsName];
    const QVariant& defl = socket.info.defaultValue;
    var = ApiUtil::qVarToStdVar(defl);
    return var;
}

void ZNode_Impl::setSocketDefl(const std::string& sockName, const ZVARIANT& value)
{
    QString qsName = QString::fromStdString(sockName);
    INPUT_SOCKETS inputs = m_index.data(ROLE_INPUTS).value<INPUT_SOCKETS>();
    if (inputs.find(qsName) == inputs.end())
        return;

    INPUT_SOCKET socket = inputs[qsName];
}

ZVARIANT ZNode_Impl::getParam(const std::string& name)
{
    ZVARIANT var;
    return var;
}

void ZNode_Impl::setParamValue(const std::string& name, const ZVARIANT& value)
{

}