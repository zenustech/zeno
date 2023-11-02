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
    ToolTip { id: toolTip; timeout: 2500 }
    function notifyUser(message) { toolTip.text=message; toolTip.open() }

    graph: Qan.Graph {
        id: graph
        Qan.VisualConnector {
            id: customConnector
            graph: graphView.graph                    // SAMPLE: When configuring  a custom visual connector, VisualConnector.graph must be set manually
            topMargin: graph.connector.height + 15
            connectorItem: Control {
                parent: customConnector
                anchors.fill: parent
                hoverEnabled: true
                visible: false                      // SAMPLE: Do not forget to hide the custom connector item by default, visual connector will set visible to true on demand
                ToolTip.visible: hovered &&
                                 ( !customConnector.connectorDragged || state === "HILIGHT" )
                onStateChanged: {
                    ToolTip.text = ( state === "HILIGHT" ? "Drop to connect" : "Drag on a target node" )
                }
                states: [
                    State { name: "NORMAL"; PropertyChanges { target: customConnectorItem; scale: 1.0 } },
                    State { name: "HILIGHT"; PropertyChanges { target: customConnectorItem; scale: 1.7 } }
                ]
                transitions: [
                    Transition { from: "NORMAL"; to: "HILIGHT"; PropertyAnimation { target: customConnectorItem; properties: "borderWidth, scale"; duration: 100 } },
                    Transition { from: "HILIGHT"; to: "NORMAL"; PropertyAnimation { target: customConnectorItem; properties: "borderWidth, scale"; duration: 150 } }
                ]
                Image {
                    anchors.fill: parent
                    id: customConnectorItem
                    source: "qrc:/fa_link.png"
                    state: "NORMAL"; smooth: true;   antialiasing: true
                }
            }
            createDefaultEdge: false    // SAMPLE: When createDefaultEdge is set to false, VisualConnector does not use Qan.Graph.insertEdge()
                                        // to create edge, but instead emit requestEdgeCreation (see below) to allow user to create custom
                                        // edge (either specifying a custom edge component, or calling a user defined method on graph).
            onRequestEdgeCreation: {
                if (src && dst && dstPortItem) {
                    notifyUser("Edge creation requested between " + src.label + " and " + dstPortItem.label)
                    let e = graph.insertEdge(src, dst);
                    graph.bindEdgeDestination(e, dstPortItem)
                } else
                    notifyUser("Edge creation requested between " + src.label + " and " + dst.label)
            }
        }
        connectorEnabled: true
        connectorColor: Material.accent
        connectorEdgeColor: Material.accent
        connectorItem : Control {
            anchors.fill: parent
            hoverEnabled: true
            visible: false              // SAMPLE: Do not forget to hide the custom connector item by default, visual connector will set visible to true on demand
            ToolTip.visible: hovered &&
                             ( !parent.connectorDragged || state === "HILIGHT" )
            onStateChanged: {
                ToolTip.text = ( state === "HILIGHT" ? "Drop to connect" : "Drag on a target node" )
            }
            states: [
                State { name: "NORMAL"; PropertyChanges { target: flag; scale: 1.0 } },
                State { name: "HILIGHT"; PropertyChanges { target: flag; scale: 1.7 } }
            ]
            transitions: [
                Transition { from: "NORMAL"; to: "HILIGHT"; PropertyAnimation { target: flag; properties: "borderWidth, scale"; duration: 100 } },
                Transition { from: "HILIGHT"; to: "NORMAL"; PropertyAnimation { target: flag; properties: "borderWidth, scale"; duration: 150 } }
            ]
            Image {
                anchors.fill: parent
                id: flag
                source: "qrc:/fa_flag.png"
                state: "NORMAL"; smooth: true;   antialiasing: true
            }
        }
        Component.onCompleted: {
            let d1 = graph.insertNode()
            d1.label = "D1"; d1.item.x = 250; d1.item.y = 50
            let d2 = graph.insertNode()
            d2.label = "D2"; d2.item.x = 250; d2.item.y = 150

            let s1 = graph.insertNode()
            s1.label = "S1"; s1.item.x = 15; s1.item.y = 85

            graph.insertEdge(s1, d1)
            graph.insertEdge(s1, d2)

            let d3 = graph.insertNode()
            d3.label = "D3"; d3.item.x = 250; d3.item.y = 250
            let d3p1 = graph.insertPort(d3, Qan.NodeItem.Left);
            d3p1.label = "D3 IN #1"

            graph.setConnectorSource(s1)
            customConnector.sourceNode = s1
        }
        onNodeClicked: {
            if (node && node.item) {
                customConnector.sourceNode = node
            } else
                customConnector.visible = false
        }
    }
    Frame {
        anchors.top: parent.top; anchors.right: parent.right; anchors.rightMargin: 10
        ColumnLayout {
            CheckBox {
                text: qsTr("Enabled Visual Connector")
                checked: graph.connectorEnabled
                onClicked: graph.connectorEnabled = checked
            }
        }
    }
}  // Qan.GraphView

