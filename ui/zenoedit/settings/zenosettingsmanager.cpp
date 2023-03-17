#include "zenosettingsmanager.h"
#include "settings/zsettings.h"


ZenoSettingsManager& ZenoSettingsManager::GetInstance()
{
    static ZenoSettingsManager instance;
    return instance;
}

ZenoSettingsManager::ZenoSettingsManager(QObject *parent) : 
    QObject(parent)
{
    QSettings settings(zsCompanyName, zsEditor);
    if (settings.allKeys().indexOf(zsShowGrid) == -1) {
        //show grid by default.
        setValue(zsShowGrid, true);
    }
}

void ZenoSettingsManager::setValue(const QString& name, const QVariant& value) 
{
    QSettings settings(zsCompanyName, zsEditor);
    QVariant oldValue = settings.value(name);
    if (oldValue != value)
    {
        settings.setValue(name, value);
        emit valueChanged(name);
    }
}

QVariant ZenoSettingsManager::getValue(const QString& zsName) const
{
    QSettings settings(zsCompanyName, zsEditor);
    QVariant val = settings.value(zsName);
    return val;
}
