/*
 Copyright (c) 2008-2021, Benoit AUTHEMAN All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the author or Destrat.io nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL AUTHOR BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


//-----------------------------------------------------------------------------
// This file is a part of the QuickQanava software library.
//
// \file	qanPlugin.cpp
// \author	alexander @machinekoder
// \date	2018 08 19
//-----------------------------------------------------------------------------

// QuickQanava headers
#include "./qanPlugin.h"
#include "./qanEdge.h"
#include "./qanEdgeItem.h"
#include "./qanNode.h"
#include "./qanNodeItem.h"
#include "./qanPortItem.h"
#include "./qanConnector.h"
#include "./qanGroup.h"
#include "./qanGroupItem.h"
#include "./qanGraph.h"
#include "./qanNavigable.h"
#include "./qanGrid.h"
#include "./qanLineGrid.h"
#include "./qanGraphView.h"
#include "./qanStyle.h"
#include "./qanStyleManager.h"
#include "./qanBottomRightResizer.h"
#include "./qanNavigablePreview.h"

static void initResources()
{
#ifndef QUICKQANAVA_STATIC
    Q_INIT_RESOURCE(QuickQanava_plugin);
#endif
}

static const struct {
    const char *type;
    int major, minor;
} qmldir [] = {
    {"LineGrid", 2, 0},
    {"Edge", 2, 0},
    {"EdgeTemplate", 2, 0},
    {"Node", 2, 0},
    {"GraphView", 2, 0},
    {"Group", 2, 0},
    {"RectNodeTemplate", 2, 0},
    {"RectSolidBackground",  2, 0},
    {"RectGroupTemplate",  2, 0},
    {"CanvasNodeTemplate",  2, 0},
    {"VisualConnector",  2, 0},
    {"LabelEditor",  2, 0},
    {"SelectionItem",  2, 0},
    {"StyleListView",  2, 0},
    {"HorizontalDock",  2, 0},
    {"VerticalDock",  2, 0},
    {"Port",  2, 0},
};

void QuickQanavaPlugin::registerTypes(const char *uri)
{
    initResources();

    // @uri QuickQanava
    qmlRegisterType<qan::Node>(uri, 2, 0, "AbstractNode");
    qmlRegisterType<qan::NodeItem>(uri, 2, 0, "NodeItem");
    qmlRegisterType<qan::PortItem>(uri, 2, 0, "PortItem");
    qmlRegisterType<qan::Edge>(uri, 2, 0, "AbstractEdge");
    qmlRegisterType<qan::EdgeItem>(uri, 2, 0, "EdgeItem");
    qmlRegisterType<qan::Group>(uri, 2, 0, "AbstractGroup");
    qmlRegisterType<qan::GroupItem>(uri, 2, 0, "GroupItem");
    qmlRegisterType<qan::Connector>(uri, 2, 0, "Connector");

    qmlRegisterType<qan::Graph>(uri, 2, 0, "Graph");
    qmlRegisterType<qan::GraphView>(uri, 2, 0, "AbstractGraphView");
    qmlRegisterType<qan::Navigable>(uri, 2, 0, "Navigable");
    qmlRegisterType<qan::NavigablePreview>(uri, 2, 0, "AbstractNavigablePreview");
    
    qmlRegisterType<qan::Grid>(uri, 2, 0, "AbstractGrid");
    qmlRegisterType<qan::OrthoGrid>(uri, 2, 0, "OrthoGrid");
    qmlRegisterType<qan::LineGrid>(uri, 2, 0, "AbstractLineGrid");
    qmlRegisterType<qan::impl::GridLine>(uri, 2, 0, "GridLine");

    qmlRegisterType<qan::Style>(uri, 2, 0, "Style");
    qmlRegisterType<qan::NodeStyle>(uri, 2, 0, "NodeStyle");
    qmlRegisterType<qan::EdgeStyle>(uri, 2, 0, "EdgeStyle");
    qmlRegisterType<qan::StyleManager>(uri, 2, 0, "StyleManager");
    qmlRegisterType<qan::BottomRightResizer>(uri, 2, 0, "BottomRightResizer" );

    const QString filesLocation = fileLocation();
    for (auto i : qmldir) {
        qmlRegisterType(QUrl(filesLocation + "/" + i.type + ".qml"), uri, i.major, i.minor, i.type);
    }
}

void QuickQanavaPlugin::initializeEngine(QQmlEngine *engine, const char *uri)
{
    Q_UNUSED(uri);

    if (isLoadedFromResource())
        engine->addImportPath(QStringLiteral("qrc:/"));

    engine->rootContext()->setContextProperty( "defaultNodeStyle", QVariant::fromValue(qan::Node::style()) );
    engine->rootContext()->setContextProperty( "defaultEdgeStyle", QVariant::fromValue(qan::Edge::style()) );
    engine->rootContext()->setContextProperty( "defaultGroupStyle", QVariant::fromValue(qan::Group::style()) );
}

QString QuickQanavaPlugin::fileLocation() const
{
    if (isLoadedFromResource())
        return QStringLiteral("qrc:/QuickQanava");
    return baseUrl().toString();
}

bool QuickQanavaPlugin::isLoadedFromResource() const
{
    // If one file is missing, it will load all the files from the resource
    QFile file(baseUrl().toLocalFile() + qmldir[0].type + ".qml");
    return !file.exists();
}
