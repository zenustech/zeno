#include "zenosettingsmanager.h"


ZenoSettingsManager& ZenoSettingsManager::GetInstance() 
{
    static ZenoSettingsManager instance;
    return instance;
}

ZenoSettingsManager::ZenoSettingsManager(QObject *parent) : 
    QObject(parent),
    m_bShowGrid(true), 
    m_bSnapGrid(false) 
{
}

void ZenoSettingsManager::setValue(ValueType type, const QVariant& value) 
{
    switch (type) 
    {
    case VALUE_SHOWGRID: 
    {
        m_bShowGrid = value.toBool();
        break;
    }
    case VALUE_SNAPGRID: 
    {
        m_bSnapGrid = value.toBool();
        break;
    }
    default:
    break;
    }
    emit valueChanged(type);
}

QVariant ZenoSettingsManager::getValue(int type) const
{
    switch (type) 
    {
    case VALUE_SHOWGRID: 
    {
        return m_bShowGrid;
    }
    case VALUE_SNAPGRID: 
    {
        return m_bSnapGrid;
    }
    default: 
    {
        return QVariant();
    }
    }
}
