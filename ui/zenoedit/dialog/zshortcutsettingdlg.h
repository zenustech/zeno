#pragma once

#include <QDialog>
#include <QTableWidget>

struct ShortCutInfo;

class ZShortCutSettingDlg : public QDialog
{
    Q_OBJECT

public:
    ZShortCutSettingDlg(QWidget *parent = nullptr);
    ~ZShortCutSettingDlg();
signals:
    void layoutChangedSignal();

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

  private:
    void initUI();
    void writeShortCutInfo();
    QString getShotCutStatus(const QString &shortcut);

  private:
    QTableWidget *m_pTableWidget;
    QVector<ShortCutInfo> m_shortCutInfos;
};
