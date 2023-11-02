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

import QtQuick                   2.15
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import Qt.labs.platform          1.0    // ColorDialog

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import "." as Qan

Qan.GraphView {
    id: graphView
    anchors.fill: parent
    navigable   : true
    PinchHandler {
        target: null
        onActiveScaleChanged: {
            console.error('centroid.position=' + centroid.position)
            console.error('activeScale=' + activeScale)
            var p = centroid.position
            var f = activeScale > 1.0 ? 1. : -1.
            graphView.zoomOn(p, graphView.zoom + (f * 0.04))
        }
    }
    graph: Qan.Graph {
        id: graph
        connectorEnabled: true              // SAMPLE: This is where visual connection of node is enabled...
        Component.onCompleted: {
            var d1 = graph.insertNode()
            d1.label = "D1"; d1.item.x = 250; d1.item.y = 50
            var d2 = graph.insertNode()
            d2.label = "D2"; d2.item.x = 250; d2.item.y = 150

            var s1 = graph.insertNode()
            s1.label = "S1"; s1.item.x = 15; s1.item.y = 85

            graph.insertEdge(s1, d1)
            var e12 = graph.insertEdge(s1, d2)
            e12.label = "TEST"

            var d3 = graph.insertNode()
            d3.label = "D3"; d3.item.x = 250; d3.item.y = 250
            graph.setConnectorSource(s1)    // SAMPLE: ... and eventually configured manually on a specific node until user select another one
        }
        function getEdgeDescription(edge) {
            var edgeSrcDst = "unknown"
            if ( edge && edge.item ) {
                var edgeItem = edge.item
                if ( edgeItem.sourceItem &&
                     edgeItem.sourceItem.node )
                    edgeSrcDst = edgeItem.sourceItem.node.label
                edgeSrcDst += " -> "
                if ( edgeItem.destinationItem &&
                     edgeItem.destinationItem.node )
                    edgeSrcDst += edgeItem.destinationItem.node.label
            }
            return edgeSrcDst
        }
        onConnectorEdgeInserted: { notifyUser("Edge inserted: " + getEdgeDescription(edge)) }
        onConnectorRequestEdgeCreation: { notifyUser("Requesting Edge creation from " + src.label + " to " + ( dst ? dst.label : "UNDEFINED" ) ) }
        onEdgeClicked: { notifyUser("Edge " + edge.label + " " + getEdgeDescription(edge) + " clicked") }
        onEdgeDoubleClicked: { notifyUser("Edge " + edge.label + " " + getEdgeDescription(edge) + " double clicked") }
        onEdgeRightClicked: { notifyUser("Edge " + edge.label + " " + getEdgeDescription(edge) + " right clicked") }
    }
    ToolTip { id: toolTip; timeout: 2500 }
    function notifyUser(message) { toolTip.text=message; toolTip.open() }

    ColorDialog {
        id: connectorEdgeColorDialog
        color: graph.connectorEdgeColor
        onAccepted: graph.connectorEdgeColor = color
    }
    ColorDialog {
        id: connectorColorDialog
        color: graph.connectorColor
        onAccepted: graph.connectorColor = color
    }
    Frame {
        anchors.top: parent.top; anchors.right: parent.right; anchors.rightMargin: 10
        ColumnLayout {
            CheckBox {
                text: qsTr("Enabled Visual Connector")
                checked: graph.connectorEnabled
                onClicked: graph.connectorEnabled = checked
            }
            CheckBox {
                text: qsTr("Create Default Edge")
                checked: graph.connectorCreateDefaultEdge
                onClicked: graph.connectorCreateDefaultEdge = checked
            }
            RowLayout {
                Rectangle { width:32; height: 32; color: graph.connectorEdgeColor; radius: 5; border.width:1; border.color: Qt.darker(color) }
                Label { text: "Connector Edge Color:" }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "..."
                    onClicked: { connectorEdgeColorDialog.open() }
                }
            }
            RowLayout {
                Rectangle { width:32; height: 32; color: graph.connectorColor; radius: 5; border.width:1; border.color: Qt.darker(color) }
                Label { text: "Connector Color:" }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "..."
                    onClicked: { connectorColorDialog.open() }
                }
            }
        }
    }
}  // Qan.GraphView

