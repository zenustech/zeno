#include "winlayoutrw.h"
#include "../dock/ztabdockwidget.h"
#include <zenoui/util/jsonhelper.h>
#include "../dock/docktabcontent.h"
#include "../viewport/viewportwidget.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include <rapidjson/document.h>



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

static void _writeLayout(PtrLayoutNode root, PRETTY_WRITER& writer)
{
    JsonObjBatch scope(writer);
    if (root->type == NT_HOR || root->type == NT_VERT)
    {
        writer.Key("orientation");
        writer.String(root->type == NT_HOR ? "H" : "V");
        writer.Key("left");
        if (root->pLeft)
            _writeLayout(root->pLeft, writer);
        else
            writer.Null();

        writer.Key("right");
        if (root->pRight)
            _writeLayout(root->pRight, writer);
        else
            writer.Null();
    }
    else
    {
        writer.Key("widget");
        if (root->pWidget == nullptr)
        {
            writer.Null();
        }
        else
        {
            writer.StartObject();

            writer.Key("geometry");
            writer.StartObject();
            QRect rc = root->pWidget->geometry();
            writer.Key("x");
            writer.Int(rc.left());
            writer.Key("y");
            writer.Int(rc.top());
            writer.Key("width");
            writer.Int(rc.width());
            writer.Key("height");
            writer.Int(rc.height());
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
                else if (qobject_cast<DisplayWidget*>(wid)) {
                    writer.String("View");
                }
                else if (qobject_cast<ZenoSpreadsheet*>(wid)) {
                    writer.String("Data");
                }
                else if (qobject_cast<ZlogPanel*>(wid)) {
                    writer.String("Logger");
                }
            }
            writer.EndArray();

            writer.EndObject();
        }
    }
}

QString exportLayout(PtrLayoutNode root)
{
    rapidjson::StringBuffer s;
    PRETTY_WRITER writer(s);
    _writeLayout(root, writer);
    QString strJson = QString::fromUtf8(s.GetString());
    return strJson;
}

void writeLayout(PtrLayoutNode root, const QString &filePath)
{
    QFile f(filePath);
    if (!f.open(QIODevice::WriteOnly)) {
        return;
    }
    QString strJson = exportLayout(root);
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
        int x = geomObj["x"].GetInt();
        int y = geomObj["y"].GetInt();
        int width = geomObj["width"].GetInt();
        int height = geomObj["height"].GetInt();
        ptrNode->geom = QRect(x, y, width, height);
        
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

