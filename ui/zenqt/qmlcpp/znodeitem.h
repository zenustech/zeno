#ifndef ZNODEITEM_H
#define ZNODEITEM_H

#include <QObject>
#include <QQuickItem>
#include <QQmlListProperty>
#include "zparamitem.h"

class ZNodeItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QString name READ getName WRITE setName NOTIFY name_changed)

public:
    ZNodeItem();
    QString getName();
    void setName(QString name);

signals:
    void name_changed();

private:
    QString m_name;
};

#endif