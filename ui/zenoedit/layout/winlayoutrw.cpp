#include "winlayoutrw.h"
#include "../dock/ztabdockwidget.h"
#include "../dock/docktabcontent.h"
#include "../viewport/viewportwidget.h"
#include "../panel/zenospreadsheet.h"
#include "../panel/zlogpanel.h"
#include <rapidjson/document.h>
#include "../panel/zenolights.h"
#include "viewport/displaywidget.h"


PtrLayoutNode findNode(PtrLayoutNode root, QDockWidget* pWidget)
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

PtrLayoutNode findParent(PtrLayoutNode root, QDockWidget* pWidget)
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

QString exportLayout(PtrLayoutNode root, const QSize& szMainwin)
{
    rapidjson::StringBuffer s;
    PRETTY_WRITER writer(s);
    ZsgWriter::getInstance()._writeLayout(root, szMainwin, writer, &AppHelper::dumpTabsToZsg);
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

    return Zsg2Reader::getInstance()._readLayout(doc.GetObject());
}


PtrLayoutNode readLayout(const QString& content)
{
    rapidjson::Document doc;
    QByteArray bytes = content.toUtf8();
    doc.Parse(bytes);
    return Zsg2Reader::getInstance()._readLayout(doc.GetObject());
}

PtrLayoutNode readLayout(const rapidjson::Value& objValue)
{
    return Zsg2Reader::getInstance()._readLayout(objValue);
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

