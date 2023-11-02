/*
 Copyright (c) 2008-2017, Benoit AUTHEMAN All rights reserved.

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

import QtQuick                   2.8
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import "." as Qan

Qan.GraphView {
    id: graphView
    anchors.fill: parent
    navigable   : true
    resizeHandlerColor: "#03a9f4"       // SAMPLE: Set resize handler color to blue for 'resizable' nodes
    gridThickColor: Material.theme === Material.Dark ? "#4e4e4e" : "#c1c1c1"

    graph: Qan.Graph {
        id: graph
        Component.onCompleted: {
            var n1 = graph.insertNode()
            n1.label = "Hello World"; n1.item.x=15; n1.item.y= 25
            n1.item.ratio = 0.4
            var n2 = graph.insertNode()
            n2.label = "Node 2"; n2.item.x=15; n2.item.y= 125

            var e = graph.insertEdge(n1, n2);
            defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
        }
        onNodeClicked: {
            notifyUser( "Node <b>" + node.label + "</b> clicked" )
            nodeEditor.node = node
        }
        onNodeRightClicked: { notifyUser( "Node <b>" + node.label + "</b> right clicked" ) }
        onNodeDoubleClicked: { notifyUser( "Node <b>" + node.label + "</b> double clicked" ) }
    }
    ToolTip { id: toolTip; timeout: 2500 }
    function notifyUser(message) { toolTip.text=message; toolTip.open() }
    Label {
        anchors.left: parent.left; anchors.leftMargin: 15
        anchors.bottom: parent.bottom; anchors.bottomMargin: 15
        text: "Use CTRL+Click to select multiples nodes"
    }
    Frame {
        id: nodeEditor
        property var node: undefined
        onNodeChanged: nodeItem = node ? node.item : undefined
        property var nodeItem: undefined
        anchors.bottom: parent.bottom; anchors.bottomMargin: 15
        anchors.right: parent.right; anchors.rightMargin: 15
        ColumnLayout {
            Label {
                text: nodeEditor.node ? "Editing node <b>" + nodeEditor.node.label + "</b>": "Select a node..."
            }
            CheckBox {
                text: "Draggable"
                enabled: nodeEditor.nodeItem !== undefined
                checked: nodeEditor.nodeItem ? nodeEditor.nodeItem.draggable : false
                onClicked: nodeEditor.nodeItem.draggable = checked
            }
            CheckBox {
                text: "Resizable"
                enabled: nodeEditor.nodeItem !== undefined
                checked: nodeEditor.nodeItem ? nodeEditor.nodeItem.resizable : false
                onClicked: nodeEditor.nodeItem.resizable = checked
            }
            CheckBox {
                text: "Selected (read-only)"
                enabled: false
                checked: nodeEditor.nodeItem ? nodeEditor.nodeItem.selected : false
            }
            CheckBox {
                text: "Selectable"
                enabled: nodeEditor.nodeItem != null
                checked: nodeEditor.nodeItem ? nodeEditor.nodeItem.selectable : false
                onClicked: nodeEditor.nodeItem.selectable = checked
            }
            Label { text: "style.backRadius" }
            Slider {
                from: 0.; to: 15.0;
                value: defaultNodeStyle.backRadius
                stepSize: 1.0
                onMoved: defaultNodeStyle.backRadius = value
            }
        }
    }
}  // Qan.GraphView

