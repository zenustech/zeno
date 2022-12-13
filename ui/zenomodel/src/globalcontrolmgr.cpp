#include "globalcontrolmgr.h"
#include "uihelper.h"
#include "zassert.h"
#include "iparammodel.h"
#include "modelrole.h"


static CONTROL_INFO _infos[] = {
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
    for (int i = 0; i < sizeof(_infos) / sizeof(CONTROL_INFO); i++)
    {
        m_infos.push_back(_infos[i]);
    }
}

void GlobalControlMgr::onParamUpdated(const QString& nodeCls, PARAM_CLASS cls, const QString& coreParam, PARAM_CONTROL newCtrl)
{
    CONTROL_INFO info(nodeCls, cls, coreParam, CONTROL_NONE, QVariant());
    int idx = m_infos.indexOf(info);
    ZASSERT_EXIT(idx != -1);
    m_infos[idx].control = newCtrl;
}

CONTROL_INFO GlobalControlMgr::controlInfo(const QString& nodeCls, PARAM_CLASS cls, const QString& coreParam, const QString& coreType)
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
        return CONTROL_INFO(CONTROL_ENUM, QVariant::fromValue(items));
    }
    return CONTROL_INFO(UiHelper::getControlByType(coreType), QVariant());
}

void GlobalControlMgr::onCoreParamsInserted(const QModelIndex &parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex& idx = pModel->index(first, 0, parent);
    QPersistentModelIndex nodeIdx = pModel->data(idx, ROLE_NODE_IDX).toPersistentModelIndex();
    const QString& objCls = nodeIdx.data(ROLE_OBJNAME).toString();
    PARAM_CLASS cls = pModel->paramClass();
    const QString& coreParam = idx.data(ROLE_PARAM_NAME).toString();

    CONTROL_INFO info(objCls, cls, coreParam, CONTROL_NONE, QVariant());
    if (m_infos.indexOf(info) == -1)
        m_infos.append(info);
}

void GlobalControlMgr::onCoreParamsAboutToBeRemoved(const QModelIndex &parent, int first, int last)
{
    IParamModel* pModel = qobject_cast<IParamModel*>(sender());
    ZASSERT_EXIT(pModel);
    const QModelIndex &idx = pModel->index(first, 0, parent);
    QPersistentModelIndex nodeIdx = pModel->data(idx, ROLE_NODE_IDX).toPersistentModelIndex();
    const QString& objCls = nodeIdx.data(ROLE_OBJNAME).toString();
    PARAM_CLASS cls = pModel->paramClass();
    const QString& coreParam = idx.data(ROLE_PARAM_NAME).toString();

    CONTROL_INFO info(objCls, cls, coreParam, CONTROL_NONE, QVariant());
    m_infos.removeAll(info);
}

void GlobalControlMgr::onSubGraphRename(const QString& oldName, const QString& newName)
{
    for (int i = 0; i < m_infos.size(); i++)
    {
        if (m_infos[i].objCls == oldName)
        {
            m_infos[i].objCls = newName;
        }
    }
}

void GlobalControlMgr::onParamRename(const QString& nodeCls, PARAM_CLASS cls, const QString& oldName, const QString& newName)
{
    CONTROL_INFO info(nodeCls, cls, oldName, CONTROL_NONE, QVariant());
    int idx = m_infos.indexOf(info);
    if (idx != -1)
    {
        m_infos[idx].coreParam = newName;
    }
}
