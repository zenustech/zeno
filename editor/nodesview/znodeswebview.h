#ifndef __ZNODES_WEBVIEW_H__
#define __ZNODES_WEBVIEW_H__

#include <QWebEngineView>

class ZNodesWebEngineView : public QWebEngineView
{
    Q_OBJECT
public:
    ZNodesWebEngineView(QWidget* parent = nullptr);

public slots:
    void reload();
};


#endif