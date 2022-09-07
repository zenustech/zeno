#ifndef __API_UTIL_H__
#define __API_UTIL_H__

#include <QVariant>
#include "enum.h"

class ApiUtil
{
public:
    static ZVARIANT qVarToStdVar(const QVariant& qvar, QString typeDesc = "");
    static QVariant stdVarToQVar(const ZVARIANT& var);
};

#endif