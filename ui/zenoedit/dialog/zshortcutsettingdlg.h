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
    void initShortCutInfos();

  private:
    QTableWidget *m_pTableWidget;
    ShortCutInfo *m_shortCutInfos;
};
