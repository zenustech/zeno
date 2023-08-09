#ifndef __WIN_LAYOUT_RW_H__
#define __WIN_LAYOUT_RW_H__

#include <memory>
#include <QtWidgets>
#include <zenomodel/include/jsonhelper.h>
#include "../dock/docktabcontent.h"

struct LayerOutNode;
class ZTabDockWidget;

typedef std::shared_ptr<LayerOutNode> PtrLayoutNode;

enum NodeType
{
    NT_HOR,
    NT_VERT,
    NT_ELEM,
};

struct LayerOutNode {
    PtrLayoutNode pLeft;
    PtrLayoutNode pRight;
    NodeType type;
    ZTabDockWidget *pWidget;
    QStringList tabs;
    QRectF geom;
    QVector<DockContentWidgetInfo> widgetInfos;
};

PtrLayoutNode findNode(PtrLayoutNode root, ZTabDockWidget *pWidget);
PtrLayoutNode findParent(PtrLayoutNode root, ZTabDockWidget *pWidget);
void writeLayout(PtrLayoutNode root, const QSize& szMainwin, const QString& filePath);
void writeLayout(PtrLayoutNode root, const QSize& szMainwin, RAPIDJSON_WRITER& writer);
QString exportLayout(PtrLayoutNode root, const QSize& szMainwin);
PtrLayoutNode readLayoutFile(const QString& filePath);
PtrLayoutNode readLayout(const QString& content);
PtrLayoutNode readLayout(const rapidjson::Value& objValue);
int getDockSize(PtrLayoutNode root, bool bHori);

#endif