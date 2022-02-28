#include <QApplication>
#include "style/zenostyle.h"
#include <comctrl/ziconbutton.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"
#include <comctrl/ztoolbutton.h>



int main(int argc, char *argv[]) 
{
	ZenoApplication a(argc, argv);
	a.setStyle(new ZenoStyle);

	QPalette palette = a.palette();
	palette.setColor(QPalette::Window, QColor(11, 11, 11));
	palette.setColor(QPalette::WindowText, Qt::white);
	a.setPalette(palette);

    //ZToolButton* btn = new ZToolButton(
    //    ZToolButton::Opt_HasIcon | ZToolButton::Opt_HasText | ZToolButton::Opt_UpRight,
    //    QIcon(":/icons/subnetbtn.svg"),
    //    QSize(28, 28),
    //    "Subset",
    //    nullptr
    //);
    //btn->show();

	ZenoMainWindow mainWindow;
	mainWindow.showMaximized();
	return a.exec();
}
