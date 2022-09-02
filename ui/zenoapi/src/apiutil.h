#ifndef __API_UTIL_H__
#define __API_UTIL_H__

#include "interface.h"

#include <QVariant>

class ApiUtil
{
public:
    static ZVARIANT qVarToStdVar(const QVariant& qvar);
    static QVariant stdVarToQVar(const ZVARIANT& var);
};

#endif