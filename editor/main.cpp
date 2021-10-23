#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //a.setAttribute(Qt::AA_EnableHighDpiScaling);
    MainWindow w;
    w.show();
    return a.exec();
}
