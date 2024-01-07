#ifndef __ZABOUTDLG_H__
#define __ZABOUTDLG_H__

namespace Ui
{
    class AboutDlg;
}

#include <QtWidgets>

class ZAboutDlg : public QDialog
{
    Q_OBJECT
public:
    ZAboutDlg(QWidget* parent = nullptr);

private:
	Ui::AboutDlg* m_ui;
};

#endif