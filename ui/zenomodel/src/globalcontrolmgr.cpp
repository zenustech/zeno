#include "globalcontrolmgr.h"
#include "uihelper.h"


struct _CONTROL_INFO
{
    QString objCls;
    PARAM_CLASS cls;
    QString coreParam;
    PARAM_CONTROL control;
    QVariant controlProps;
};

static _CONTROL_INFO _infos[] = {
    {"ParticleParticleWrangle",     PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"ParticlesWrangle",            PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"VDBWrangle",                  PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"NumericWrangle",              PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"ParticlesMaskedWrangle",      PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"ParticlesNeighborBvhWrangle", PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"ParticlesTwoWrangle",         PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"StringEval",                  PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"NumericEval",                 PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"ParticlesNeighborz",          PARAM_INPUT, "zfxCode",     CONTROL_MULTILINE_STRING, QVariant()},
    {"SubInput",                    PARAM_PARAM, "type",        CONTROL_ENUM,             QVariant::fromValue(UiHelper::getCoreTypeList())},
};



GlobalControlMgr& GlobalControlMgr::instance()
{
    static GlobalControlMgr mgr;
    return mgr;
}

GlobalControlMgr::GlobalControlMgr()
{

}

CONTROL_INFO GlobalControlMgr::controlInfo(const QString& objCls, PARAM_CLASS cls, const QString& coreParam, const QString& coreType)
{
    for (int i = 0; i < sizeof(_infos) / sizeof(_CONTROL_INFO); i++)
    {
        if (_infos[i].objCls == objCls && _infos[i].cls == cls && _infos[i].coreParam == coreParam)
        {
            return CONTROL_INFO(_infos[i].control, _infos[i].controlProps);
        }
    }
    if (coreType.startsWith("enum "))
    {
        QStringList items = coreType.mid(QString("enum ").length()).split(QRegExp("\\s+"));
        return CONTROL_INFO(CONTROL_ENUM, QVariant::fromValue(items));
    }
    return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
}