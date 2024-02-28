#ifndef __WIN_LAYOUT_RW_H__
#define __WIN_LAYOUT_RW_H__

#include <memory>
#include <QtWidgets>
#include <zenomodel/include/jsonhelper.h>
#include "../dock/docktabcontent.h"
#include <zeno/io/zsgwriter.h>
#include <zeno/io/zsg2reader.h>
//#include "util/apphelper.h"

struct LayerOutNode;
typedef std::shared_ptr<LayerOutNode> PtrLayoutNode;

enum LayoutNodeType
{
    NT_HOR,
    NT_VERT,
    NT_ELEM,
};

struct DockContentWidgetInfo {
    int resolutionX;
    int resolutionY;
    bool lock;
    int comboboxindex;
    double colorR;
    double colorG;
    double colorB;
    DockContentWidgetInfo(int resX, int resY, bool block, int index, double r, double g, double b) : resolutionX(resX), resolutionY(resY), lock(block), comboboxindex(index), colorR(r), colorG(g), colorB(b) {}
    DockContentWidgetInfo(int resX, int resY, bool block, int index) : resolutionX(resX), resolutionY(resY), lock(block), comboboxindex(index) {}
    DockContentWidgetInfo() : resolutionX(0), resolutionY(0), lock(false), comboboxindex(0), colorR(0.18f), colorG(0.20f), colorB(0.22f) {}
};

struct LayerOutNode {
    PtrLayoutNode pLeft;
    PtrLayoutNode pRight;
    LayoutNodeType type;
    QDockWidget* pWidget;
    QStringList tabs;
    QRectF geom;
    QVector<DockContentWidgetInfo> widgetInfos;
};

PtrLayoutNode findNode(PtrLayoutNode root, QDockWidget*pWidget);
PtrLayoutNode findParent(PtrLayoutNode root, QDockWidget*pWidget);
void writeLayout(PtrLayoutNode root, const QSize& szMainwin, const QString& filePath);
QString exportLayout(PtrLayoutNode root, const QSize& szMainwin);
PtrLayoutNode readLayoutFile(const QString& filePath);
PtrLayoutNode readLayout(const QString& content);
PtrLayoutNode readLayout(const rapidjson::Value& objValue);
int getDockSize(PtrLayoutNode root, bool bHori);
PtrLayoutNode _readLayout(const rapidjson::Value& objValue);

#endif