#ifndef ZPARAMITEM_H
#define ZPARAMITEM_H

#include <QObject>
#include <QQuickItem>

class ZParamItem : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QString name READ getName WRITE setName NOTIFY name_changed)

    QML_ELEMENT

public:
    ZParamItem();
    QString getName();
    void setName(QString name);

signals:
    void name_changed();

private:
    QString m_name;
};

#endif