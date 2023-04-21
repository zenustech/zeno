#ifndef __ZENOSETTINGS_MANAGR__
#define __ZENOSETTINGS_MANAGER__

#include <QtWidgets>
#include "settings/zsettings.h"


struct ShortCutInfo {
    QString key;
    QString desc;
    QString shortcut;
};
Q_DECLARE_METATYPE(ShortCutInfo)

class ZenoSettingsManager : public QObject
{
    Q_OBJECT
public:
    static ZenoSettingsManager& GetInstance();
    void setValue(const QString& name, const QVariant& value);
    QVariant getValue(const QString& zsName) const;

    const int getShortCut(const QString &key);
    void setShortCut(const QString &key, const QString &value);
    void writeShortCutInfo(const QVector<ShortCutInfo> &infos);

signals:
    void valueChanged(QString zsName);

private:
    void initShortCutInfos();
    QVector<ShortCutInfo> getDefaultShortCutInfo();
    int getShortCutInfo(const QString &key, ShortCutInfo &info);

private:
    ZenoSettingsManager(QObject *parent = nullptr);
    QVector<ShortCutInfo> m_shortCutInfos;
};


#endif