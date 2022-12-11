#ifndef __GLOBAL_CONTROL_MGR_H__
#define __GLOBAL_CONTROL_MGR_H__

#include "modeldata.h"
#include "modelrole.h"

struct CONTROL_INFO
{
    PARAM_CONTROL ctrl;
    QVariant props;
    CONTROL_INFO(PARAM_CONTROL ctrl, QVariant props) : ctrl(ctrl), props(props) {}
};

class GlobalControlMgr
{
public:
    static GlobalControlMgr& instance();
    CONTROL_INFO controlInfo(const QString& objCls, PARAM_CLASS cls, const QString& coreParam, const QString& coreType);

private:
    GlobalControlMgr();
};

#endif