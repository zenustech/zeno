#include "globalcontrolmgr.h"
#include "uihelper.h"
#include "uicommon.h"


static CONTROL_INFO _infos[] = {
    {"SubInput",  true, "type", zeno::Combobox, QVariant()},
    {"SubOutput", true, "type", zeno::Combobox, QVariant()},
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

CONTROL_INFO GlobalControlMgr::controlInfo(const QString& nodeCls, bool bInput, const QString& coreParam, const QString& coreType) const
{
    CONTROL_INFO info(nodeCls, true, coreParam, zeno::NullControl, QVariant());
    int idx = m_infos.indexOf(info);
    if (idx != -1)
    {
        if (m_infos[idx].control == zeno::NullControl)
            return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
        return m_infos[idx];
    }
    if (coreType.startsWith("enum "))
    {
        QStringList items = coreType.mid(QString("enum ").length()).split(QRegExp("\\s+"));
        QVariantMap map;
        map["items"] = items;
        return CONTROL_INFO(zeno::Combobox, map);
    }
    if (coreParam == "zfxCode" && coreType == "string" ||
        coreParam == "commands" && coreType == "string")
    {
        return CONTROL_INFO(zeno::Multiline, QVariant());
    }
    if (nodeCls == "PythonNode" && coreParam == "script")
    {
        return CONTROL_INFO(zeno::Multiline, QVariant());
    }
    return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
}
