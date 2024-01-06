#include "globalcontrolmgr.h"
#include "uihelper.h"
#include "zassert.h"
#include "nodeparammodel.h"
#include "modelrole.h"
#include "vparamitem.h"


static CONTROL_INFO _infos[] = {
    {"SubInput",                    PARAM_PARAM, "type",        CONTROL_ENUM,             QVariant()},
    {"SubOutput", PARAM_PARAM, "type", CONTROL_ENUM, QVariant()},
};



GlobalControlMgr& GlobalControlMgr::instance()
{
    static GlobalControlMgr mgr;
    return mgr;
}

GlobalControlMgr::GlobalControlMgr()
{
    for (int i = 0; i < sizeof(_infos) / sizeof(CONTROL_INFO); i++)
    {
        if (_infos[i].objCls == "SubInput" || _infos[i].objCls == "SubOutput")
        {
            QVariantMap map;
            map["items"] = UiHelper::getCoreTypeList();
            _infos[i].controlProps = map;
        }
        m_infos.push_back(_infos[i]);
    }
}

CONTROL_INFO GlobalControlMgr::controlInfo(const QString& nodeCls, PARAM_CLASS cls, const QString& coreParam, const QString& coreType) const
{
    CONTROL_INFO info(nodeCls, cls, coreParam, CONTROL_NONE, QVariant());
    int idx = m_infos.indexOf(info);
    if (idx != -1)
    {
        if (m_infos[idx].control == CONTROL_NONE)
            return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
        return m_infos[idx];
    }
    if (coreType.startsWith("enum "))
    {
        QStringList items = coreType.mid(QString("enum ").length()).split(QRegExp("\\s+"));
        QVariantMap map;
        map["items"] = items;
        return CONTROL_INFO(CONTROL_ENUM, map);
    }
    if (coreParam == "zfxCode" && coreType == "string" ||
        coreParam == "commands" && coreType == "string")
    {
        return CONTROL_INFO(CONTROL_MULTILINE_STRING, QVariant());
    }
    if (nodeCls == "GenerateCommands" && coreParam == "source")
    {
        return CONTROL_INFO(CONTROL_BUTTON, QVariant());
    }
    if (nodeCls == "PythonNode" && coreParam == "script")
    {
        return CONTROL_INFO(CONTROL_MULTILINE_STRING, QVariant());
    }
    return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
}
