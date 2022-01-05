#include "mainwindow.h"
#include <QApplication>
#include <kddockwidgets/DockWidget.h>
#include <kddockwidgets/MainWindow.h>
#include <kddockwidgets/Config.h>
#include "tmpwidgets/zmainwindow.h"
#include "style/zenostyle.h"
#include <comctrl/ziconbutton.h>
#include <nodesys/zenosearchbar.h>
#include "zenoapplication.h"
#include "zenomainwindow.h"

//#define TEST_WEBENGINE
//#define TEST_SEARCHBAR
//#define TEST_ICONBUTTON
//#define USE_KKDOCK
#define USE_NEWWIN

ZENO_NAMESPACE_BEGIN

int zenoMainWithKDDoc(int argc, char* argv[])
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    KDDockWidgets::Config::self().setAbsoluteWidgetMinSize(QSize(54, 54));

    ZMainWindow mainWindow;
    mainWindow.showMaximized();
    return a.exec();
}

int zenoMainWithNewWin(int argc, char* argv[])
{
    ZenoApplication a(argc, argv);
    a.setStyle(new ZenoStyle);

    QPalette palette = a.palette();
    palette.setColor(QPalette::Window, QColor(11, 11, 11));
    palette.setColor(QPalette::WindowText, Qt::white);
    a.setPalette(palette);

    ZenoMainWindow mainWindow;
    mainWindow.showMaximized();
    return a.exec();
}

int zenoMain(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //a.setStyle("Fusion");
    a.setStyle(new ZenoStyle);

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

#ifdef TEST_SEARCHBAR
    QApplication app(argc, argv);
    ZenoSearchBar *searcher = new ZenoSearchBar;
    searcher->show();
    return app.exec();
#endif

#ifdef TEST_ICONBUTTON
    QApplication app(argc, argv);
    ZIconButton *pBtn = new ZIconButton(QIcon(":/icons/search_arrow.svg"), QSize(32, 32), QColor(30, 30, 30), QColor(0, 0, 0), false);
    pBtn->show();
    return app.exec();
#endif

#ifdef TEST_WEBENGINE
    #include <QWebEngineView>

    QApplication app(argc, argv);
    QWebEngineView view;
    view.load(QUrl("https://zenustech.com/"));
    view.show();
    return app.exec();
#elif defined(USE_NEWWIN)
    return ZENO_NAMESPACE::zenoMainWithNewWin(argc, argv);
#elif defined(USE_KKDOCK)
    return ZENO_NAMESPACE::zenoMainWithKDDoc(argc, argv);
#else
    return ZENO_NAMESPACE::zenoMain(argc, argv);
#endif
}
