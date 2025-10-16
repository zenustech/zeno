#include "quick_parameter.h"


static int id_ = 0;

ZQuickParam::ZQuickParam()
    : m_bInput(true)
    , m_type(ZPARAM_STRING)
    , m_control(ZQuickParam::CTRL_NONE)
{
    //if (id_ == 0) {
    //    m_name = "sockName233";
    //    id_++;
    //}
    //else
    //    m_name = "sockName234";

    m_timer = new QTimer;
    connect(m_timer, &QTimer::timeout, this, [=]() {
        if (false) {
            setName("what the fuck");
        }
    });
    m_timer->start(1000);
}

QString ZQuickParam::getName()
{
    return m_name;
}

void ZQuickParam::setInput(bool bInput)
{
    m_bInput = bInput;
}

void ZQuickParam::setName(QString name) {
    m_name = name;
    emit name_changed();
}

PARAM_TYPE ZQuickParam::getType()
{
    return m_type;
}

void ZQuickParam::setType(PARAM_TYPE type)
{
    m_type = type;
    emit type_changed();
}

ZQuickParam::CONTROL_TYPE ZQuickParam::getControl()
{
    return m_control;
}

void ZQuickParam::setControl(CONTROL_TYPE control)
{
    m_control = control;
    emit control_changed();
}