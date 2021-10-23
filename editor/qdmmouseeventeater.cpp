#include "qdmmouseeventeater.h"

QDMMouseEventEater::QDMMouseEventEater(QObject *parent) : QObject(parent)
{

}

bool QDMMouseEventEater::eventFilter(QObject *object, QEvent *event)
{
    return false;
    return QObject::eventFilter(object, event);
}
