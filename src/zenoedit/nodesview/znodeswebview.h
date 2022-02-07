#ifndef __ZNODES_WEBVIEW_H__
#define __ZNODES_WEBVIEW_H__

#include <QtWidgets>

    class ZNodesWebEngineView : public QWidget
    {
        Q_OBJECT
    public:
        ZNodesWebEngineView(QWidget* parent = nullptr);
    public slots:
        void reload();
    };

#endif

