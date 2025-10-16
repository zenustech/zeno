#include "zparamitem.h"


ZParamItem::ZParamItem()
{

}

QString ZParamItem::getName()
{
    return m_name;
}

void ZParamItem::setName(QString name)
{
    m_name = name;
    emit name_changed();
}