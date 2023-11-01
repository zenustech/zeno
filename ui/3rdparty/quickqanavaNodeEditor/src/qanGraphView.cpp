/*
 Copyright (c) 2008-2022, Benoit AUTHEMAN All rights reserved.

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
// \file	qanGraphView.cpp
// \author	benoit@destrat.io
// \date	2016 08 15
//-----------------------------------------------------------------------------

// Qt headers
#include <QQuickItem>

// QuickQanava headers
#include "./qanNavigable.h"
#include "./qanGraphView.h"
#include "./qanGraph.h"

namespace qan { // ::qan

/* GraphView Object Management *///--------------------------------------------
GraphView::GraphView(QQuickItem* parent) :
    qan::Navigable{parent}
{
    setAntialiasing(true);
    setSmooth(true);
    setFocus(true);
}

void    GraphView::setGraph(qan::Graph* graph)
{
    if (graph == nullptr) {
        qWarning() << "qan::GraphView::setGraph(): Error: Setting a nullptr graph in Qan.GraphView is not supported.";
        return;
    }
    if (graph != _graph) {
        if (_graph != nullptr)
            disconnect(_graph, 0, this, 0);
        _graph = graph;
        auto graphViewQmlContext = qmlContext(this);
        QQmlEngine::setContextForObject(getContainerItem(), graphViewQmlContext);
        _graph->setContainerItem(getContainerItem());
        connect(_graph, &qan::Graph::nodeClicked,
                this,   &qan::GraphView::nodeClicked);

        connect(_graph, &qan::Graph::connectorChanged,
                this,   &qan::GraphView::connectorChanged);

        connect(_graph, &qan::Graph::nodeRightClicked,
                this,   &qan::GraphView::nodeRightClicked);
        connect(_graph, &qan::Graph::nodeDoubleClicked,
                this,   &qan::GraphView::nodeDoubleClicked);

        connect(_graph, &qan::Graph::portClicked,
                this,   &qan::GraphView::portClicked);
        connect(_graph, &qan::Graph::portRightClicked,
                this,   &qan::GraphView::portRightClicked);

        connect(_graph, &qan::Graph::edgeClicked,
                this,   &qan::GraphView::edgeClicked);
        connect(_graph, &qan::Graph::edgeRightClicked,
                this,   &qan::GraphView::edgeRightClicked);
        connect(_graph, &qan::Graph::edgeDoubleClicked,
                this,   &qan::GraphView::edgeDoubleClicked);

        connect(_graph, &qan::Graph::groupClicked,
                this,   &qan::GraphView::groupClicked);
        connect(_graph, &qan::Graph::groupRightClicked,
                this,   &qan::GraphView::groupRightClicked);
        connect(_graph, &qan::Graph::groupDoubleClicked,
                this,   &qan::GraphView::groupDoubleClicked);
        connect(_graph, &qan::Graph::startDrag,
                this, &qan::GraphView::startDrag);
        emit graphChanged();
    }
}

void    GraphView::navigableClicked(QPointF pos)
{
    Q_UNUSED(pos)
    if (_graph)
        _graph->clearSelection();
}

void    GraphView::navigableRightClicked(QPointF pos)
{
    emit    rightClicked(pos);
}

QString GraphView::urlToLocalFile(QUrl url) const noexcept
{
    if (url.isLocalFile())
        return url.toLocalFile();
    return QString{};
}
//-----------------------------------------------------------------------------


/* Selection Rectangle Management *///-----------------------------------------
void    GraphView::selectionRectActivated(const QRectF& rect)
{
    if (!_graph ||
        _graph->getContainerItem() == nullptr)
        return;
    if (rect.isEmpty())
        return;
    // Algorithm:
    // 1. Iterate over all already selected items, remove one that are no longer inside selection rect
    //    (for example, if selection rect has grown down...)
    // 2. Iterate over all graph items, select items inside selection rect.

    // 1.
    QSetIterator<QQuickItem*> selectedItem(_selectedItems);
    while (selectedItem.hasNext()) {
        const auto item = selectedItem.next();
        auto nodeItem = qobject_cast<qan::NodeItem*>(item);
        if (nodeItem != nullptr &&
            nodeItem->getNode() != nullptr ) {
            const auto itemBr = item->mapRectToItem(_graph->getContainerItem(),
                                                    item->boundingRect());
            if (!rect.intersects(itemBr)) {
                _graph->setNodeSelected(*nodeItem->getNode(), false);
                _selectedItems.remove(item);
            }
        }
        auto edgeItem = qobject_cast<qan::EdgeItem*>(item);
        if (edgeItem != nullptr &&
            edgeItem->getEdge() != nullptr) {
            if (!edgeItem->selected(rect)) {
                _graph->selectEdge(*edgeItem->getEdge(), Qt::ControlModifier);
                _selectedItems.remove(item);
            }
        }
    }

    // 2.
    const auto items = _graph->getContainerItem()->childItems();
    for (const auto item: items) {
        auto nodeItem = qobject_cast<qan::NodeItem*>(item);
        if (nodeItem != nullptr &&
            nodeItem->getNode() != nullptr) {
            auto node = nodeItem->getNode();
            const auto itemBr = item->mapRectToItem(_graph->getContainerItem(),
                                                    item->boundingRect());
            if (rect.intersects(itemBr)) {
                _graph->setNodeSelected(*node, true);
                // Note we assume that items are not deleted while the selection
                // is in progress... (QPointer can't be trivially inserted in QSet)
                _selectedItems.insert(nodeItem);
            }
        }
        auto edgeItem = qobject_cast<qan::EdgeItem*>(item);
        if (edgeItem != nullptr &&
            edgeItem->getEdge() != nullptr) {
            if (edgeItem->selected(rect)) {
                if (edgeItem->getSelected())
                    continue;
                _graph->selectEdge(*edgeItem->getEdge(), Qt::ControlModifier);
                _selectedItems.insert(edgeItem);
            }
        }
    }
}

void    GraphView::selectionRectEnd()
{
    _selectedItems.clear();  // Clear selection cache
}
//-----------------------------------------------------------------------------

} // ::qan

