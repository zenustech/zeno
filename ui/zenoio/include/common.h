#ifndef __IO_COMMON_H__
#define __IO_COMMON_H__

namespace zenoio {
    enum ZSG_VERSION
    {
        VER_2,          //old version io
        VER_2_5,        //new version io
    };
}

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

struct LayerOutNode;
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
    QDockWidget* pWidget;
    QStringList tabs;
    QRectF geom;
    QVector<DockContentWidgetInfo> widgetInfos;
};

#endif