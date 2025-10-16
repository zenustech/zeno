#include "znodeitem.h"


ZNodeItem::ZNodeItem()
{
}

QString ZNodeItem::getName()
{
    return m_name;
}

void ZNodeItem::setName(QString name)
{
    m_name = name;
    emit name_changed();
}