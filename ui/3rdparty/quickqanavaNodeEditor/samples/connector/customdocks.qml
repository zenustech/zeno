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
import Qt.labs.platform          1.0    // ColorDialog

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import "." as Qan


Qan.GraphView {
    id: graphView
    z: -1
    anchors.fill: parent
    navigable: true

    graph: Qan.Graph {
        id: topology
        anchors.fill: parent
        objectName: "graph"
        clip: true
        connectorEnabled: false

        nodeDelegate: Component {
            Qan.NodeItem {
                width: 150
                height: 80
                leftDock: Qan.VerticalDock {
                    Label {
                        Layout.preferredWidth: height
                        text: "Custom Dock"
                        font.bold: true
                        rotation: -90
                    }
                }
                Rectangle {
                    anchors.fill: parent
                    color: "lightblue"
                    radius: 5
                    border.color: "green"; border.width: 4
                    Label { anchors.centerIn: parent; text: "CUSTOM" }
                }
            }
        }

        portDelegate: Component {
            id: youpi
            Qan.PortItem {
                width: 16; height: 16
                Rectangle {
                    anchors.fill: parent
                    color: "grey"
                    border.color: "yellow"; border.width: 2
                }
            }
        }

        Component.onCompleted: {
            var n1 = topology.insertNode(nodeDelegate)
            n1.label = "Default.Node"
            n1.item.x = 30
            n1.item.y = 30
            topology.insertPort(n1, Qan.NodeItem.Left)
            topology.insertPort(n1, Qan.NodeItem.Left)
            topology.insertPort(n1, Qan.NodeItem.Right)
            topology.insertPort(n1, Qan.NodeItem.Right)
        }
    }
}


