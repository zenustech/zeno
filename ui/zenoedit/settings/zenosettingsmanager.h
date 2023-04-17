#ifndef __ZENOSETTINGS_MANAGR__
#define __ZENOSETTINGS_MANAGER__

#include <QVariant>
#include <QObject>
#include <QSettings>
#include "settings/zsettings.h"

struct ShortCutInfo {
    QString key;
    QString desc;
    QString shortcut;
    ShortCutInfo *next = nullptr;
    ~ShortCutInfo() {
        delete next;
        next = nullptr;
    }
    void clone(const ShortCutInfo *other) {
        key = other->key;
        desc = other->desc;
        shortcut = other->shortcut;
        if (other->next) {
            next = new ShortCutInfo();
            next->clone(other->next);
        }
    }
};
Q_DECLARE_METATYPE(ShortCutInfo*)

class ZenoSettingsManager : public QObject
{
    Q_OBJECT
public:
    static ZenoSettingsManager& GetInstance();
    void setValue(const QString& name, const QVariant& value);
    QVariant getValue(const QString& zsName) const;

    const int getShortCut(const QString &key);
    void setShortCut(const QString &key, const QString &value);

    void writeShortCutInfo(const ShortCutInfo*infos);

  signals:
    void valueChanged(QString zsName);

private:
    void initShortCutInfos();
    void getDefaultShortCutInfo(ShortCutInfo **info);
    ShortCutInfo *getShortCutInfo(const QString &key, ShortCutInfo *info);

  private:
    ZenoSettingsManager(QObject *parent = nullptr);
    ShortCutInfo *m_shortCutInfos;
};


#endif