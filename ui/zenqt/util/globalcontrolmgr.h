#ifndef __GLOBAL_CONTROL_MGR_H__
#define __GLOBAL_CONTROL_MGR_H__

#include "uicommon.h"
#include <zeno/core/data.h>

struct CONTROL_INFO {
    QString objCls;
    bool bInput;
    QString coreParam;
    zeno::ParamControl control;
    QVariant controlProps;

    CONTROL_INFO()
        : bInput(false)
        , control(zeno::NullControl)
    {
    }

    CONTROL_INFO(zeno::ParamControl control, QVariant controlProps)
        : control(control)
        , controlProps(controlProps)
        , bInput(false)
    {
    }

    CONTROL_INFO(QString objCls, bool input, QString coreParam, zeno::ParamControl control, QVariant controlProps)
        : objCls(objCls)
        , coreParam(coreParam)
        , bInput(input)
        , control(control)
        , controlProps(controlProps)
    {
    }

    bool operator==(const CONTROL_INFO& rInfo) const
    {
        return objCls == rInfo.objCls && bInput == rInfo.bInput && coreParam == rInfo.coreParam;
    }
};

class GlobalControlMgr : public QObject
{
    Q_OBJECT
public:
    static GlobalControlMgr& instance();
    CONTROL_INFO controlInfo(const QString& objCls, bool bInput, const QString& coreParam, const QString& coreType) const;

private:
    GlobalControlMgr();

    QVector<CONTROL_INFO> m_infos;
};

#endif