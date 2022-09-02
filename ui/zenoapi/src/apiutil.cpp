#include "apiutil.h"


ZVARIANT ApiUtil::qVarToStdVar(const QVariant& qvar)
{
    ZVARIANT var;
    switch (qvar.type())
    {
    case QVariant::Int:
        var = qvar.toInt();
        break;
    case QVariant::Bool:
        var = qvar.toBool();
        break;
    case QMetaType::Float:
        var = qvar.toFloat();
        break;
    case QVariant::Double:
        var = qvar.toDouble();
        break;
    case QVariant::String:
        var = qvar.toString().toStdString();
        break;
    case QVariant::UserType:
        //todo:
        break;
    default:
        break;
    };
    return var;
}

QVariant ApiUtil::stdVarToQVar(const ZVARIANT& var)
{
    return QVariant();
}