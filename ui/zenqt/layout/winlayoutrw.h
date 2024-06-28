#ifndef __WIN_LAYOUT_RW_H__
#define __WIN_LAYOUT_RW_H__

#include <memory>
#include <QtWidgets>
#include <util/jsonhelper.h>
#include "layout/docktabcontent.h"
#include <zeno/io/zsg2reader.h>


struct LayerOutNode;
typedef std::shared_ptr<LayerOutNode> PtrLayoutNode;

namespace ads
{
    class CDockManager;
}

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
    ads::CDockWidget* pWidget;
    QStringList tabs;
    QRectF geom;
    QVector<DockContentWidgetInfo> widgetInfos;
};

PtrLayoutNode findNode(PtrLayoutNode root, ads::CDockWidget* pWidget);
PtrLayoutNode findParent(PtrLayoutNode root, ads::CDockWidget* pWidget);
void writeLayout(PtrLayoutNode root, const QSize& szMainwin, const QString& filePath);
QString exportLayout(PtrLayoutNode root, const QSize& szMainwin);
QString exportLayout(ads::CDockManager *pManager);
PtrLayoutNode readLayoutFile(const QString& filePath);
PtrLayoutNode readLayout(const QString& content);
bool readLayout(const QString& content, QString& state, QStringList& widgets);
PtrLayoutNode readLayout(const rapidjson::Value& objValue);
int getDockSize(PtrLayoutNode root, bool bHori);
PtrLayoutNode _readLayout(const rapidjson::Value& objValue);

#endif