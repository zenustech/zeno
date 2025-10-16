#ifndef QUICK_PARAMETER_H
#define QUICK_PARAMETER_H

#include <QObject>
#include <QQuickItem>
#include <QVariant>
#include <QTimer>
#include <qqml.h>
#include "uicommon.h"

//需要定义一个control类型，并且能在qml也能复用

class ZQuickParam : public QQuickItem
{
    Q_OBJECT
        //Q_PROPERTY(bool input READ isInput)
    Q_PROPERTY(QString name READ getName WRITE setName NOTIFY name_changed)
    Q_PROPERTY(bool input READ isInput WRITE setInput)

public:
    enum CONTROL_TYPE
    {
        CTRL_NONE,
        CTRL_LINEEDIT,
        CTRL_MULTITEXT,
        CTRL_PATH,
        CTRL_COMBOBOX,
        CTRL_VEC2,
        CTRL_VEC3,
        CTRL_VEC4,
        CTRL_CHECKBOX
    };
    Q_ENUMS(CONTROL_TYPE)

private:
    //Q_PROPERTY(PARAM_TYPE type READ getType WRITE setType NOTIFY type_changed)
    
    Q_PROPERTY(CONTROL_TYPE control READ getControl WRITE setControl NOTIFY control_changed)

    QML_ELEMENT

public:
    ZQuickParam();
    bool isInput() const { return m_bInput; }
    void setInput(bool bInput);
    QString getName();
    void setName(QString name);
    PARAM_TYPE getType();
    void setType(PARAM_TYPE type);
    CONTROL_TYPE getControl();
    void setControl(CONTROL_TYPE control);

signals:
    void name_changed();
    void type_changed();
    void control_changed();

private:
    QString         m_name;
    bool            m_bInput;
    PARAM_TYPE      m_type;
    CONTROL_TYPE    m_control;
    QVariant        m_value;
    QTimer* m_timer;
};

#endif