#include "mainwindow.h"
#include <QApplication>
#include <kddockwidgets/DockWidget.h>
#include <kddockwidgets/MainWindow.h>
#include <kddockwidgets/Config.h>

#define USE_KKDOCK

ZENO_NAMESPACE_BEGIN

int zenoMainWithKDDoc(int argc, char* argv[])
{
    QApplication a(argc, argv);

    KDDockWidgets::Config::self().setAbsoluteWidgetMinSize(QSize(54, 54));
    KDDockWidgets::MainWindow mainWindow(QStringLiteral("MyMainWindow"));
	mainWindow.setWindowTitle("Main Window");
	mainWindow.resize(1200, 1200);
	mainWindow.show();

	auto dock1 = new KDDockWidgets::DockWidget(QStringLiteral("MyDock1"));
	auto widget1 = new QWidget();
	dock1->setWidget(widget1);

	auto dock2 = new KDDockWidgets::DockWidget(QStringLiteral("MyDock2"));
	auto widget2 = new QWidget();
	dock2->setWidget(widget2);

	mainWindow.addDockWidget(dock1, KDDockWidgets::Location_OnLeft);
	mainWindow.addDockWidget(dock2, KDDockWidgets::Location_OnTop);

    return a.exec();
}

int zenoMain(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //a.setStyle("Fusion");

    QPalette palette;
    palette.setColor(QPalette::Window, QColor(53, 53, 53));
    palette.setColor(QPalette::WindowText, Qt::white);
    palette.setColor(QPalette::Base, QColor(25, 25, 25));
    palette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    palette.setColor(QPalette::ToolTipBase, Qt::black);
    palette.setColor(QPalette::ToolTipText, Qt::white);
    palette.setColor(QPalette::Text, Qt::white);
    palette.setColor(QPalette::Button, QColor(53, 53, 53));
    palette.setColor(QPalette::ButtonText, Qt::white);
    palette.setColor(QPalette::BrightText, Qt::red);
    palette.setColor(QPalette::Link, QColor(42, 130, 218));
    palette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    palette.setColor(QPalette::HighlightedText, Qt::black);
    a.setPalette(palette);

    MainWindow w;
    w.show();
    return a.exec();
}

ZENO_NAMESPACE_END

int main(int argc, char *argv[]) {
#ifdef USE_KKDOCK
    return ZENO_NAMESPACE::zenoMainWithKDDoc(argc, argv);
#else
    return ZENO_NAMESPACE::zenoMain(argc, argv);
#endif
}
