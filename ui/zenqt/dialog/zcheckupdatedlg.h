#ifndef __ZCHECKUPDATEDLG_H__
#define __ZCHECKUPDATEDLG_H__

namespace Ui
{
    class ZCheckUpdateDlg;
}

#include "dialog/zframelessdialog.h"

class ZCheckUpdateDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZCheckUpdateDlg(QWidget* parent = nullptr);
signals:
    void updateSignal(const QString& version, const QString& url);
    void remindSignal();

private slots:
    void slt_netReqFinish(const QString& data, const QString& id);

private:
    void getCudaVersion();
    void requestLatestVersion();
    void initUI();
    void initConnection();
    void updateView(bool bShow);

private:
	Ui::ZCheckUpdateDlg* m_ui;
    QString m_url;
    QString m_version;
    QString m_cudaVersion;
};

#endif