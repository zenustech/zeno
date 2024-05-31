#ifndef __ZSOCKET_SETTING_DLG_H__
#define __ZSOCKET_SETTING_DLG_H__

#include "dialog/zframelessdialog.h"

class ZSocketSettingDlg : public ZFramelessDialog
{
    Q_OBJECT

public:
    ZSocketSettingDlg(const QModelIndexList& indexs, QWidget *parent = nullptr);
    ~ZSocketSettingDlg();

private slots:
    void onOKClicked();
private:
    void initView();
    void initButtons();

private:
    QModelIndexList m_indexs;
    QWidget* m_mainWidget;
    QStandardItemModel* m_pModel;
};
#endif
