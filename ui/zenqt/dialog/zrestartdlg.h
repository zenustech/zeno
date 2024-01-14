#ifndef __ZRESTARTDLG_H__
#define __ZRESTARTDLG_H__

namespace Ui
{
    class ZRestartDlg;
}

#include "dialog/zframelessdialog.h"

class ZRestartDlg : public ZFramelessDialog
{
    Q_OBJECT
public:
    ZRestartDlg(QWidget* parent = nullptr);
signals:
    void saveSignal(bool bSaveAs);

private:
	Ui::ZRestartDlg* m_ui;
    QString m_version;
};

#endif