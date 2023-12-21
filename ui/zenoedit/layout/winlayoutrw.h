#ifndef __WIN_LAYOUT_RW_H__
#define __WIN_LAYOUT_RW_H__

#include <memory>
#include <QtWidgets>
#include <zenomodel/include/jsonhelper.h>
#include "../dock/docktabcontent.h"
#include <zenoio/writer/zsgwriter.h>
#include <zenoio/reader/zsg2reader.h>
#include "util/apphelper.h"

PtrLayoutNode findNode(PtrLayoutNode root, QDockWidget*pWidget);
PtrLayoutNode findParent(PtrLayoutNode root, QDockWidget*pWidget);
void writeLayout(PtrLayoutNode root, const QSize& szMainwin, const QString& filePath);
QString exportLayout(PtrLayoutNode root, const QSize& szMainwin);
PtrLayoutNode readLayoutFile(const QString& filePath);
PtrLayoutNode readLayout(const QString& content);
PtrLayoutNode readLayout(const rapidjson::Value& objValue);
int getDockSize(PtrLayoutNode root, bool bHori);

#endif