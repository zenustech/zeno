/*
 Copyright (c) 2008-2023, Benoit AUTHEMAN All rights reserved.

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
// \file	GraphView.qml
// \author	benoit@destrat.io
// \date	2015 08 01
//-----------------------------------------------------------------------------

import QtQuick          2.7
import QtQuick.Shapes   1.0

import QuickQanava      2.0 as Qan
import "qrc:/QuickQanava" as Qan

/*! \brief Visual view for a Qan.Graph component.
 *
 *  Set the \c graph property to a Qan.Graph{} component.
 */
Qan.AbstractGraphView {
    id: graphView

    // PUBLIC ////////////////////////////////////////////////////////////////
    //! Grid line and thick points color
    property color  gridThickColor: grid ? grid.thickColor : lineGrid.thickColor

    //! Visual selection rectangle (CTRL + right click drag) color, default to Material blue.
    property color  selectionRectColor: Qt.rgba(0.129, 0.588, 0.953, 1) // Material blue: #2196f3

    property color  resizeHandlerColor: Qt.rgba(0.117, 0.564, 1.0)  // dodgerblue=rgb( 30, 144, 255)
    property real   resizeHandlerOpacity: 1.0
    property real   resizeHandlerRadius: 4.0
    property real   resizeHandlerWidth: 4.0
    property size   resizeHandlerSize: "9x9"

    // PRIVATE ////////////////////////////////////////////////////////////////
    Qan.LineGrid {
        id: lineGrid
    }
    grid: lineGrid
    onGridThickColorChanged: {
        if (grid)
            grid.thickColor = gridThickColor
    }

    Qan.BottomRightResizer {
        id: nodeResizer
        parent: graph.containerItem
        visible: false

        enabled: target && target.node && target.node.commitStatus !== 2    // Disable for locked nodes
        opacity: resizeHandlerOpacity
        handlerColor: resizeHandlerColor
        handlerRadius: resizeHandlerRadius
        handlerWidth: resizeHandlerWidth
        handlerSize: resizeHandlerSize
        onResizeStart: {
            if (target && target.node)
                graph.nodeAboutToBeResized(target.node);
        }
        onResizeEnd: {
            if (target && target.node)
                graph.nodeResized(target.node);
        }
    }
    Qan.RightResizer {
        id: nodeRightResizer
        parent: graph.containerItem

        enabled: target && target.node && target.node.commitStatus !== 2
        onResizeStart: {
            if (target && target.node)
                graph.nodeAboutToBeResized(target.node);
        }
        onResizeEnd: {
            if (target && target.node)
                graph.nodeResized(target.node);
        }
    }
    Qan.BottomResizer {
        id: nodeBottomResizer
        parent: graph.containerItem
        enabled: target && target.node && target.node.commitStatus !== 2
        onResizeStart: {
            if (target && target.node)
                graph.nodeAboutToBeResized(target.node);
        }
        onResizeEnd: {
            if (target && target.node)
                graph.nodeResized(target.node);
        }
    }
    Qan.BottomRightResizer {
        id: groupResizer
        parent: graph.containerItem
        visible: false
        enabled: target && target.node && target.node.commitStatus !== 2    // Disable for locked nodes

        opacity: resizeHandlerOpacity
        handlerColor: resizeHandlerColor
        handlerRadius: resizeHandlerRadius
        handlerWidth: resizeHandlerWidth
        handlerSize: resizeHandlerSize

        onResizeStart: {
            if (target && target.group)
                graph.groupAboutToBeResized(target.group)
        }
        onResizeEnd: {
            if (target && target.group)
                graph.groupResized(target.group)
        }
    }
    Qan.RightResizer {
        id: groupRightResizer
        parent: graph.containerItem
        enabled: target && target.node && target.node.commitStatus !== 2    // Disable for locked nodes
        onResizeStart: {
            if (target && target.group)
                graph.groupAboutToBeResized(target.group);
        }
        onResizeEnd: {
            if (target && target.group)
                graph.groupResized(target.group);
        }
    }
    Qan.BottomResizer {
        id: groupBottomResizer
        parent: graph.containerItem
        enabled: target && target.node && target.node.commitStatus !== 2    // Disable for locked nodes
        onResizeStart: {
            if (target && target.group)
                graph.groupAboutToBeResized(target.group);
        }
        onResizeEnd: {
            if (target && target.group)
                graph.groupResized(target.group);
        }
    }

    Rectangle {
        id: selectionRect
        x: 0; y: 0
        width: 10; height: 10
        border.width: 2
        border.color: selectionRectColor
        color: Qt.rgba(0, 0, 0, 0)  // transparent
        visible: false
    }
    selectionRectItem: selectionRect

    // View Click management //////////////////////////////////////////////////
    onClicked: {
        // Hide resizers when view background is clicked
        nodeResizer.target = nodeRightResizer.target = nodeBottomResizer.target = null
        groupResizer.target = groupRightResizer.target = groupBottomResizer.target = null
        groupResizer.targetContent = groupRightResizer.targetContent = groupBottomResizer.targetContent = null

        // Hide the default visual edge connector
        if (graph &&
            graph.connectorEnabled &&
            graph.connector &&
            graph.connector.visible)
            graph.connector.visible = false

        graphView.focus = true           // User clicked outside a graph item, remove it's eventual active focus
    }
    onRightClicked: {
        graphView.focus = true
    }

    // Port management ////////////////////////////////////////////////////////
    onPortClicked: function(port) {
        if (graph &&
            port) {
            if (port.node)    // Force port host node on top
                graph.sendToFront(port.node.item)
            if (graph.connector &&
                graph.connectorEnabled)
                graph.connector.sourcePort = port
        } else if (graph)
            graph.connector.visible = false
    }
    onPortRightClicked: { }

    // Node management ////////////////////////////////////////////////////////

    // Dynamically handle currently selected node item onRatioChanged() signal
    Connections { // and update nodeResizer ratio policy (selected node is nodeResizer target)
        id: nodeItemRatioWatcher
        target: null
        function onRatioChanged() {
            if (nodeResizer &&
                target &&
                nodeResizer.target === target) {
                nodeResizer.preserveRatio = target.ratio > 0.
                nodeResizer.ratio = target.ratio
            }
        }
    }

    onNodeClicked: function(node) {
        if (!graphView.graph ||
            !node ||
            !node.item)
            return

        if (node.locked ||
                node.isProtected)           // Do not show any connector for locked node/groups
            return;

        graph.sendToFront(node.item)    // Protected/Locked nodes are not re-ordered to front.
        if (graph.connector &&
                graph.connectorEnabled &&
                (node.item.connectable === Qan.NodeItem.Connectable ||
                 node.item.connectable === Qan.NodeItem.OutConnectable)) {      // Do not show visual connector if node is not visually "connectable"
            graph.connector.visible = true
            graph.connector.sourceNode = node
            // Connector should be half on top of node
            graph.connector.y = -graph.connector.height / 2
        }
        if (node.item.resizable) {
            nodeItemRatioWatcher.target = node.item

            nodeResizer.minimumTargetSize = node.item.minimumSize
            nodeRightResizer.minimumTargetSize = nodeBottomResizer.minimumTargetSize = node.item.minimumSize

            nodeResizer.target = node.item
            nodeRightResizer.target = nodeBottomResizer.target = node.item

            nodeResizer.visible = nodeRightResizer.visible = nodeBottomResizer.visible =
                    Qt.binding(() => { return nodeResizer.target ? nodeResizer.target.visible && nodeResizer.target.resizable :
                                                                   false; })


            nodeResizer.z = graph.maxZ + 4    // We want resizer to stay on top of selection item and ports.
            nodeRightResizer.z = nodeBottomResizer.z = graph.maxZ + 4

            nodeResizer.preserveRatio = (node.item.ratio > 0.)
            if (node.item.ratio > 0.) {
                nodeResizer.ratio = node.item.ratio
                nodeResizer.preserveRatio = true
            } else
                nodeResizer.preserveRatio = false
        } else {
            nodeResizer.target = null
            nodeRightResizer.target = nodeBottomResizer.target = null
        }
    }

    // Group management ///////////////////////////////////////////////////////
    onGroupClicked: function(group) {
        if (!graphView.graph ||
            !group ||
            !group.item)
            return

        // Disable node resizing
        nodeResizer.target = nodeRightResizer.target = nodeBottomResizer.target = null

        if (!group.locked && !group.isProtected)  // Do not move locked/protected groups to front.
            graph.sendToFront(group.item)

        if (group.item.container &&
            group.item.resizable) {
            // Set minimumTargetSize _before_ setting target
            groupResizer.minimumTargetSize = group.item.minimumSize
            groupResizer.target = group.item
            groupResizer.targetContent = group.isTable() ? null : group.item.container
            groupRightResizer.minimumTargetSize = groupBottomResizer.minimumTargetSize = group.item.minimumSize
            groupRightResizer.target = groupBottomResizer.target = group.item
            groupRightResizer.targetContent = groupBottomResizer.targetContent = group.isTable() ? null : group.item.container

            // Do not show resizers when group is collapsed
            groupRightResizer.visible = groupBottomResizer.visible =
                    groupResizer.visible = Qt.binding(() => { // Resizer is visible :
                                                          return group && ! group.locked &&
                                                          group.item &&   // If group and group.item are valid
                                                          group.item.visible         &&
                                                          (!group.item.collapsed)    &&   // And if group is not collapsed
                                                          group.item.resizable;           // And if group is resizeable
                                                      })

            groupResizer.z = graph.maxZ + 4    // We want resizer to stay on top of selection item and ports.
            groupResizer.preserveRatio = false
            groupRightResizer.z = groupBottomResizer.z = graph.maxZ + 4
            groupRightResizer.preserveRatio = groupBottomResizer.preserveRatio = false
        } else {
            groupResizer.target = groupResizer.targetContent = null
            groupRightResizer.target = groupBottomResizer.target = null
            groupRightResizer.targetContent = groupBottomResizer.targetContent = null
        } // group.item.resizable
    }  // onGroupClicked()

    onGroupRightClicked: function(group) {
        if (group && group.item)
            graph.sendToFront(group.item)
    }
    onGroupDoubleClicked: function(group) {
        if (group && group.itm)
            graph.sendToFront(group.item)
    }
    ShaderEffectSource {        // Screenshot shader is used for gradbbing graph containerItem screenshot. Default
        id: graphImageShader    // Item.grabToImage() does not allow negative (x, y) position, ShaderEffectSource is
        visible: false          // used to render graph at desired resolution with a custom negative sourceRect, then
        enabled: false          // it is rendered to an image with ShaderEffectSource.grabToImage() !
    }

    /*! \brief Grab graph view to an image file.
     *
     * filePath must be an url.
     * Maximum zoom level is 2.0, set zoom to undefined to let default 1.0 zoom.
     */
    function    grabGraphImage(filePath, zoom, border) {
        if (!graph ||
            !graph.containerItem) {
            console.error('Qan.GraphView.onRequestGrabGraph(): Error, graph or graph container item is invalid.')
            return
        }
        let localFilePath = graphView.urlToLocalFile(filePath)
        if (!localFilePath || localFilePath === '') {
            console.error('GraphView.grapbGraphImage(): Invalid file url ' + filePath)
            return
        }
        if (!border)
            border = 0
        let origin = Qt.point(graph.containerItem.childrenRect.x, graph.containerItem.childrenRect.y)
        graph.containerItem.width = graph.containerItem.childrenRect.width - (origin.x < 0. ? origin.x : 0.)
        graph.containerItem.height = graph.containerItem.childrenRect.height - (origin.y < 0. ? origin.y : 0.)

        graphImageShader.sourceItem = graph.containerItem
        if (!zoom)
            zoom = 1.0
        if (zoom > 2.0001)
            zoom = 2.
        let border2 = 2. * border
        let imageSize = Qt.size(border2 + graph.containerItem.childrenRect.width * zoom,
                                border2 + graph.containerItem.childrenRect.height * zoom)
        graphImageShader.width = imageSize.width
        graphImageShader.height = imageSize.height
        graphImageShader.sourceRect = Qt.rect(graph.containerItem.childrenRect.x - border,
                                              graph.containerItem.childrenRect.y - border,
                                              graph.containerItem.childrenRect.width + border2,
                                              graph.containerItem.childrenRect.height + border2)

        if (!graphImageShader.grabToImage(function(result) {
                                if (!result.saveToFile(localFilePath)) {
                                    console.error('Error while writing image to ' + filePath)
                                }
                                // Reset graphImageShader
                                graphImageShader.sourceItem = undefined
                           }, imageSize))
            console.error('Qan.GraphView.onRequestGrabGraph(): Graph screenshot request failed.')
    }
} // Qan.GraphView

