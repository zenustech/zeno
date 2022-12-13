#ifndef __GLOBAL_CONTROL_MGR_H__
#define __GLOBAL_CONTROL_MGR_H__

#include "modeldata.h"
#include "modelrole.h"

struct CONTROL_INFO {
    QString objCls;
    PARAM_CLASS cls;
    QString coreParam;
    PARAM_CONTROL control;
    QVariant controlProps;

    CONTROL_INFO(PARAM_CONTROL control, QVariant controlProps)
        : control(control)
        , controlProps(controlProps)
        , cls(PARAM_INPUT)
    {
    }

    CONTROL_INFO(QString objCls, PARAM_CLASS cls, QString coreParam, PARAM_CONTROL control, QVariant controlProps)
        : objCls(objCls)
        , coreParam(coreParam)
        , cls(cls), control(control)
        , controlProps(controlProps)
    {
    }

    bool operator==(const CONTROL_INFO& rInfo) const
    {
        return objCls == rInfo.objCls && cls == rInfo.cls && coreParam == rInfo.coreParam;
    }
};

class GlobalControlMgr : public QObject
{
    Q_OBJECT
public:
    static GlobalControlMgr& instance();
    CONTROL_INFO controlInfo(const QString& objCls, PARAM_CLASS cls, const QString& coreParam, const QString& coreType);

public slots:
    void onParamUpdated(const QString &objCls, PARAM_CLASS cls, const QString &coreParam, PARAM_CONTROL newCtrl);
    void onCoreParamsInserted(const QModelIndex &parent, int first, int last);
    void onCoreParamsAboutToBeRemoved(const QModelIndex &parent, int first, int last);
    void onSubGraphRename(const QString& oldName, const QString& newName);
    void onParamRename(const QString& nodeCls, PARAM_CLASS cls, const QString &oldName, const QString &newName);

private:
    GlobalControlMgr();

    QVector<CONTROL_INFO> m_infos;
};

#endif