#ifndef __ZNODES_WEBVIEW_H__
#define __ZNODES_WEBVIEW_H__

#include <QtWidgets>

#ifdef Q_OS_LINUX

    #include <QWebEngineView>
    class ZNodesWebEngineView : public QWebEngineView
    {
        Q_OBJECT
    public:
        ZNodesWebEngineView(QWidget* parent = nullptr);

    public slots:
        void reload();
    };

#else

    class ZNodesWebEngineView : public QWidget
    {
        Q_OBJECT
    public:
        ZNodesWebEngineView(QWidget* parent = nullptr);
    public slots:
        void reload();
    };

#endif

#endif