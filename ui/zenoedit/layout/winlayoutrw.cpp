#include "winlayoutrw.h"
#include "../dock/ztabdockwidget.h"
#include <zenomodel/include/jsonhelper.h>
#include "../dock/docktabcontent.h"
#include "../viewport/viewportwidget.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include <rapidjson/document.h>
#include "../panel/zenolights.h"


PtrLayoutNode findNode(PtrLayoutNode root, ZTabDockWidget* pWidget)
{
    PtrLayoutNode pNode;
    if (root->pWidget != nullptr)
    {
        if (root->pWidget == pWidget)
            return root;
        return nullptr;
    }
    if (root->pLeft)
    {
        pNode = findNode(root->pLeft, pWidget);
        if (pNode)
            return pNode;
    }
    if (root->pRight)
    {
        pNode = findNode(root->pRight, pWidget);
        if (pNode)
            return pNode;
    }
    return nullptr;
}

PtrLayoutNode findParent(PtrLayoutNode root, ZTabDockWidget* pWidget)
{
    if ((root->pLeft && root->pLeft->pWidget == pWidget) ||
        (root->pRight && root->pRight->pWidget == pWidget))
    {
        return root;
    }
    if (root->pLeft)
    {
        PtrLayoutNode node = findParent(root->pLeft, pWidget);
        if (node)
            return node;
    }
    if (root->pRight)
    {
        PtrLayoutNode node = findParent(root->pRight, pWidget);
        if (node)
            return node;
    }
    return nullptr;
}

static void _writeLayout(PtrLayoutNode root, const QSize& szMainwin, PRETTY_WRITER& writer)
{
    JsonObjBatch scope(writer);
    if (root->type == NT_HOR || root->type == NT_VERT)
    {
        writer.Key("orientation");
        writer.String(root->type == NT_HOR ? "H" : "V");
        writer.Key("left");
        if (root->pLeft)
            _writeLayout(root->pLeft, szMainwin, writer);
        else
            writer.Null();

        writer.Key("right");
        if (root->pRight)
            _writeLayout(root->pRight, szMainwin, writer);
        else
            writer.Null();
    }
    else
    {
        writer.Key("widget");
        if (root->pWidget == nullptr || root->pWidget->isHidden())
        {
            writer.Null();
        }
        else
        {
            writer.StartObject();
            int w = szMainwin.width();
            int h = szMainwin.height();
            if (w == 0)
                w = 1;
            if (h == 0)
                h = 1;

            writer.Key("geometry");
            writer.StartObject();
            QRect rc = root->pWidget->geometry();

            writer.Key("x");
            float _left = (float)rc.left() / w;
            writer.Double(_left);

            writer.Key("y");
            float _top = (float)rc.top() / h;
            writer.Double(_top);

            writer.Key("width");
            float _width = (float)rc.width() / w;
            writer.Double(_width);

            writer.Key("height");
            float _height = (float)rc.height() / h;
            writer.Double(_height);

            writer.EndObject();

            writer.Key("tabs");
            writer.StartArray();
            for (int i = 0; i < root->pWidget->count(); i++)
            {
                QWidget* wid = root->pWidget->widget(i);
                if (qobject_cast<DockContent_Parameter*>(wid)) {
                    writer.String("Parameter");
                }
                else if (qobject_cast<DockContent_Editor *>(wid)) {
                    writer.String("Editor");
                }
                else if (qobject_cast<DockContent_View*>(wid)) {
                    writer.String("View");
                }
                else if (qobject_cast<ZenoSpreadsheet*>(wid)) {
                    writer.String("Data");
                }
                else if (qobject_cast<DockContent_Log*>(wid)) {
                    writer.String("Logger");
                }
                else if (qobject_cast<ZenoLights*>(wid)) {
                    writer.String("Light");
                }
            }
            writer.EndArray();

            writer.EndObject();
        }
    }
}

QString exportLayout(PtrLayoutNode root, const QSize& szMainwin)
{
    rapidjson::StringBuffer s;
    PRETTY_WRITER writer(s);
    _writeLayout(root, szMainwin, writer);
    QString strJson = QString::fromUtf8(s.GetString());
    return strJson;
}

void writeLayout(PtrLayoutNode root, const QSize& szMainwin, const QString &filePath)
{
    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly)) {
        return;
    }
    QString strJson = exportLayout(root, szMainwin);
    f.write(strJson.toUtf8());
}

static PtrLayoutNode _readLayout(const rapidjson::Value& objValue)
{
    if (objValue.HasMember("orientation") && objValue.HasMember("left") && objValue.HasMember("right"))
    {
        PtrLayoutNode ptrNode = std::make_shared<LayerOutNode>();
        QString ori = objValue["orientation"].GetString();
        ptrNode->type = (ori == "H" ? NT_HOR : NT_VERT);
        ptrNode->pLeft = _readLayout(objValue["left"]);
        ptrNode->pRight = _readLayout(objValue["right"]);
        ptrNode->pWidget = nullptr;
        return ptrNode;
    }
    else if (objValue.HasMember("widget"))
    {
        PtrLayoutNode ptrNode = std::make_shared<LayerOutNode>();
        ptrNode->type = NT_ELEM;
        ptrNode->pLeft = nullptr;
        ptrNode->pRight = nullptr;

        const rapidjson::Value& widObj = objValue["widget"];
        
        auto tabsObj = widObj["tabs"].GetArray();
        QStringList tabs;
        for (int i = 0; i < tabsObj.Size(); i++)
        {
            ptrNode->tabs.push_back(tabsObj[i].GetString());
        }

        const rapidjson::Value& geomObj = widObj["geometry"];
        float x = geomObj["x"].GetFloat();
        float y = geomObj["y"].GetFloat();
        float width = geomObj["width"].GetFloat();
        float height = geomObj["height"].GetFloat();
        ptrNode->geom = QRectF(x, y, width, height);

        return ptrNode;
    }
    else
    {
        return nullptr;
    }
}

PtrLayoutNode readLayoutFile(const QString& filePath)
{
    QFile file(filePath);
    bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
    if (!ret) {
        return nullptr;
    }

    rapidjson::Document doc;
    QByteArray bytes = file.readAll();
    doc.Parse(bytes);

    return _readLayout(doc.GetObject());
}


PtrLayoutNode readLayout(const QString& content)
{
    rapidjson::Document doc;
    QByteArray bytes = content.toUtf8();
    doc.Parse(bytes);
    return _readLayout(doc.GetObject());
}

int getDockSize(PtrLayoutNode root, bool bHori)
{
    if (!root)
        return 0;

    if (root->type == NT_ELEM)
    {
        return bHori ? root->geom.width() : root->geom.height();
    }
    else if (root->type == NT_HOR)
    {
        if (bHori) {
            return getDockSize(root->pLeft, true) + 4 + getDockSize(root->pRight, true);
        }
        else {
            return getDockSize(root->pLeft, false);
        }
    }
    else if (root->type == NT_VERT)
    {
        if (bHori) {
            return getDockSize(root->pLeft, true);
        }
        else {
            return getDockSize(root->pLeft, false) + 4 + getDockSize(root->pRight, false);
        }
    }
    else
    {
        return 0;
    }
}

