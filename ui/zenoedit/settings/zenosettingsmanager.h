#ifndef __ZENOSETTINGS_MANAGR__
#define __ZENOSETTINGS_MANAGER__

#include <QVariant>
#include <QObject>
#include <QSettings>
#include "settings/zsettings.h"


class ZenoSettingsManager : public QObject
{
    Q_OBJECT
public:
    static ZenoSettingsManager& GetInstance();
    void setValue(const QString& name, const QVariant& value);
    QVariant getValue(const QString& zsName) const;
signals:
    void valueChanged(QString zsName);

private:
    ZenoSettingsManager(QObject *parent = nullptr);
};


#endif