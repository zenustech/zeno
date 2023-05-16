#include "customuimgr.h"


CustomUIMgr::CustomUIMgr()
{

}

CustomUIMgr& CustomUIMgr::instance()
{
    static CustomUIMgr inst;
    return inst;
}

void CustomUIMgr::setNodeParamXml(const QString& nodeCls, const QString& xml)
{
    m_customParams[nodeCls] = xml;
}

QString CustomUIMgr::nodeParamXml(const QString& nodeCls)
{
    return m_customParams[nodeCls];
}
