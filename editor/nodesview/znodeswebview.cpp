#include "znodeswebview.h"

#ifdef Q_OS_LINUX

ZNodesWebEngineView::ZNodesWebEngineView(QWidget* parent)
    : QWebEngineView(parent)
{
    load(QUrl::fromLocalFile("/home/luzh/zeno/editor/nodesview/viewport.html"));
}

void ZNodesWebEngineView::reload()
{
    load(QUrl::fromLocalFile("/home/luzh/zeno/editor/nodesview/viewport.html"));
}

#else

ZNodesWebEngineView::ZNodesWebEngineView(QWidget* parent)
    : QWidget(parent)
{
}

void ZNodesWebEngineView::reload()
{
}

#endif