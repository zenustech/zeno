#ifndef __ZENOSETTINGS_MANAGR__
#define __ZENOSETTINGS_MANAGER__

#include <QVariant>
#include <QObject>

class ZenoSettingsManager : public QObject
{
    Q_OBJECT
public:
    enum ValueType {
    VALUE_SHOWGRID = 0,
    VALUE_SNAPGRID
    };
    static ZenoSettingsManager& GetInstance();
    void setValue(ValueType type, const QVariant& value);
    QVariant getValue(int type) const;
signals:
    void valueChanged(int type);

  private:
    ZenoSettingsManager(QObject *parent = nullptr);

private:
    bool m_bShowGrid;
    bool m_bSnapGrid;
};


#endif