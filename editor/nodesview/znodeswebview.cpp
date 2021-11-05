#include "znodeswebview.h"


ZNodesWebEngineView::ZNodesWebEngineView(QWidget* parent)
    : QWebEngineView(parent)
{
    load(QUrl::fromLocalFile("/home/luzh/zeno/editor/nodesview/viewport.html"));
}

void ZNodesWebEngineView::reload()
{
    load(QUrl::fromLocalFile("/home/luzh/zeno/editor/nodesview/viewport.html"));
}