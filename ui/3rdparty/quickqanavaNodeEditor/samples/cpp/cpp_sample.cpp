/*
 Copyright (c) 2008-2020, Benoit AUTHEMAN All rights reserved.

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
// \file	cpp_sample.cpp
// \author	benoit@destrat.io
// \date	2018 05 24
//-----------------------------------------------------------------------------

// Qt headers
#include <QGuiApplication>
#include <QtQml>
#include <QQuickStyle>

// QuickQanava headers
#include <QuickQanava.h>

#include "./cpp_sample.h"

using namespace qan;

QQmlComponent*  CustomGroup::delegate(QQmlEngine &engine, QObject* parent) noexcept {
    Q_UNUSED(parent)
    static std::unique_ptr<QQmlComponent> delegate;
    if (!delegate)
        delegate = std::make_unique<QQmlComponent>(&engine, "qrc:/CustomGroup.qml");
    return delegate.get();
}

qan::NodeStyle* CustomGroup::style(QObject* parent) noexcept {
    Q_UNUSED(parent)
    static std::unique_ptr<qan::NodeStyle> style;
    if (!style) {
        style = std::make_unique<qan::NodeStyle>();
    }
    return style.get();
}

QQmlComponent*  CustomNode::delegate(QQmlEngine &engine, QObject* parent) noexcept
{
    Q_UNUSED(parent)
    static std::unique_ptr<QQmlComponent> customRectNode_delegate;
    if (!customRectNode_delegate)
        customRectNode_delegate =
                std::make_unique<QQmlComponent>(&engine, "qrc:/CustomNode.qml");
    return customRectNode_delegate.get();
}

qan::NodeStyle *CustomNode::style(QObject* parent) noexcept
{
    Q_UNUSED(parent)
    static std::unique_ptr<qan::NodeStyle> customRectNode_style;
    if (!customRectNode_style) {
        customRectNode_style = std::make_unique<qan::NodeStyle>();
        customRectNode_style->setBackColor(QColor("#ff29fc"));
    }
    return customRectNode_style.get();
}

QQmlComponent *CustomEdge::delegate(QQmlEngine &engine, QObject* parent) noexcept {
    Q_UNUSED(parent)
    static std::unique_ptr<QQmlComponent> customEdge_delegate;
    if (!customEdge_delegate)
        customEdge_delegate =
                std::make_unique<QQmlComponent>(&engine, "qrc:/CustomEdge.qml");
    return customEdge_delegate.get();
}

qan::EdgeStyle *CustomEdge::style(QObject* parent) noexcept {
    Q_UNUSED(parent)
    static std::unique_ptr<qan::EdgeStyle> customEdge_style;
    if (!customEdge_style)
        customEdge_style = std::make_unique<qan::EdgeStyle>();
    return customEdge_style.get();
}

qan::Group *CustomGraph::insertCustomGroup() {
  return qan::Graph::insertGroup<CustomGroup>();
}

qan::Node *CustomGraph::insertCustomNode() {
  return qan::Graph::insertNode<CustomNode>(nullptr);
}

qan::Edge *CustomGraph::insertCustomEdge(qan::Node *source,
                                         qan::Node *destination) {
  const auto engine = qmlEngine(this);
  if (source != nullptr &&
      destination != nullptr &&
      engine != nullptr)
    return qan::Graph::insertEdge<CustomEdge>(*source, destination, CustomEdge::delegate(*engine));
  return nullptr;
}


//-----------------------------------------------------------------------------
int	main( int argc, char** argv )
{
    QGuiApplication app(argc, argv);
    QQuickStyle::setStyle("Material");
    QQmlApplicationEngine engine;
    engine.addPluginPath(QStringLiteral("../../src")); // Necessary only for development when plugin is not installed to QTDIR/qml
    QuickQanava::initialize(&engine);

    qmlRegisterType<CustomGroup>("MyModule", 1, 0, "CustomGroup");
    qmlRegisterType<CustomNode>("MyModule", 1, 0, "CustomNode");
    qmlRegisterType<CustomGraph>("MyModule", 1, 0, "CustomGraph");
    qmlRegisterType<CustomEdge>("MyModule", 1, 0, "AbstractCustomEdge");

    engine.load(QUrl("qrc:/cpp_sample.qml"));

    { // We can here customize QuickQanava graph topology _synchronously_ before the
      // Qt/QML event loop starts
        QPointer<CustomGraph> graph = nullptr;
        for (const auto rootObject : engine.rootObjects()) {
            graph = qobject_cast<CustomGraph*>(rootObject->findChild<QQuickItem *>("graph"));
            if (graph)
                break;
        }

        if (graph) {
            static constexpr qreal defaultWidth{40.}, defaultHeight{30.};
            static constexpr qreal xSpacing{50.}, ySpacing{30.};

            static constexpr int array_size = 3;
            qan::Node* nodes[array_size][array_size];
            for (int r = 0; r < array_size; r++) {
                qreal nodeX = r * (defaultWidth + 2 * xSpacing );
                for (int c = 0; c < array_size; c++) {
                    qreal nodeY = c * (defaultHeight + 2 * ySpacing);
                    auto node = graph->insertCustomNode();
                    node->getItem()->setMinimumSize({defaultWidth / 2., defaultHeight / 2.});
                    node->getItem()->setRect({nodeX, nodeY, defaultWidth, defaultHeight});
                    // Equivalent to:
                        //node->getItem()->setX(nodeX);
                        //node->getItem()->setY(nodeY);
                        //node->getItem()->setWidth(defaultWidth);
                        //node->getItem()->setHeight(defaultHeight);
                        //node->setLabel(QString::number(n++));
                    nodes[r][c] = node;
                } // for columns
            } // for rows
            auto group = graph->insertCustomGroup();
            group->getItem()->setRect({300, 200, 400, 250});

            // Grouping nodes from c++
            graph->groupNode(group, nodes[0][0], nullptr, false);

            // NOTE: If the node is already on top of the group were we want to insert node, use transformPosition=true
            // to convert node position from graphview coordinate system to group coordinate system.
            // If the transformPosition=false is used, node will be grouped at (O, O) position of group, and
            // it's position should be set after the call to groupNode().
        }
    }

    return app.exec();
}
//-----------------------------------------------------------------------------


