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

import QtQuick.Window 2.2
import QtQuick 2.13

import QtQuick.Controls 2.13

import QtQuick.Layouts  1.3
import QtQuick.Controls.Material 2.1
import QtQuick.Shapes            1.0

import QuickQanava      2.0 as Qan
import "qrc:/QuickQanava" as Qan

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 720 // MPEG - 2 HD 720p - 1280 x 720 16:9
    title: "Tools test"
    Pane {
        anchors.fill: parent
        padding: 0
    }

    function centerItem(item) {
        if (!item || !window.contentItem)
            return
        var windowCenter = Qt.point(
                    (window.contentItem.width - item.width) / 2.,
                    (window.contentItem.height - item.height) / 2.)
        var graphNodeCenter = window.contentItem.mapToItem(
                    graphView.containerItem, windowCenter.x, windowCenter.y)
        item.x = graphNodeCenter.x
        item.y = graphNodeCenter.y
    }

    Qan.GraphView {
        id: graphView
        anchors.fill: parent
        graph: topology
        navigable: true
        resizeHandlerColor: Material.accent
        gridThickColor: Material.theme === Material.Dark ? "#4e4e4e" : "#c1c1c1"

        Qan.Graph {
            id: topology
            objectName: "graph"
            anchors.fill: parent
            clip: true
            connectorEnabled: true
            selectionColor: Material.accent
            connectorColor: Material.accent
            connectorEdgeColor: Material.accent
            onConnectorEdgeInserted: edge => {
                //if (edge)
                //    edge.label = "My edge"
            }
            property Component faceNodeComponent: Qt.createComponent("qrc:/FaceNode.qml")

            Component.onCompleted: {
                var n1 = topology.insertNode()
                n1.label = "n1"

                var n2 = topology.insertNode()
                n2.label = "n2"
                n2.item.x = 150
                n2.item.y = 55

                graphView.centerOnPosition(Qt.point(0, 0));
            }
        } // Qan.Graph: graph
    }

    Qan.GraphPreview {
        id: graphPreview
        source: graphView
        viewWindowColor: Material.accent
        anchors.right: graphView.right
        anchors.bottom: graphView.bottom
        anchors.rightMargin: 8
        anchors.bottomMargin: 8
        width: previewMenu.mediumPreview.width
        height: previewMenu.mediumPreview.height
        Menu {
            id: previewMenu
            readonly property size smallPreview: Qt.size(150, 85)
            readonly property size mediumPreview: Qt.size(250, 141)
            readonly property size largePreview: Qt.size(350, 198)
            MenuItem {
                text: "Hide preview"
                onTriggered: graphPreview.visible = false
            }
            MenuSeparator { }
            MenuItem {
                text: qsTr('Small')
                checkable: true
                checked: graphPreview.width === previewMenu.smallPreview.width &&
                         graphPreview.height === previewMenu.smallPreview.height
                onTriggered: {
                    graphPreview.width = previewMenu.smallPreview.width
                    graphPreview.height = previewMenu.smallPreview.height
                }
            }
            MenuItem {
                text: qsTr('Medium')
                checkable: true
                checked: graphPreview.width === previewMenu.mediumPreview.width &&
                         graphPreview.height === previewMenu.mediumPreview.height
                onTriggered: {
                    graphPreview.width = previewMenu.mediumPreview.width
                    graphPreview.height = previewMenu.mediumPreview.height
                }
            }
            MenuItem {
                text: qsTr('Large')
                checkable: true
                checked: graphPreview.width === previewMenu.largePreview.width &&
                         graphPreview.height === previewMenu.largePreview.height
                onTriggered: {
                    graphPreview.width = previewMenu.largePreview.width
                    graphPreview.height = previewMenu.largePreview.height
                }
            }
        }
        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.RightButton
            onClicked: previewMenu.open(mouse.x, mouse.y)
        }
    }

    Qan.HeatMapPreview {
        id: heatMapPreview
        anchors.left: graphView.left
        anchors.bottom: graphView.bottom
        source: graphView
        viewWindowColor: Material.accent
        Menu {
            id: heatMapMenu
            MenuItem {
                text: qsTr("Clear heat map")
                onClicked: heatMapPreview.clearHeatMap()
            }
            MenuItem {
                text: qsTr("Increase preview size")
                onTriggered: {
                    heatMapPreview.width *= 1.15
                    heatMapPreview.height *= 1.15
                }
            }
            MenuItem {
                text: qsTr("Decrease preview size")
                onTriggered: {
                    heatMapPreview.width *= Math.max(50, heatMapPreview.width * 0.85)
                    heatMapPreview.height *= Math.max(50, heatMapPreview.height * 0.85)
                }
            }
        }
        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.RightButton; preventStealing: true
            onClicked: {
                if (mouse.button === Qt.RightButton) {
                    heatMapMenu.x = mouse.x
                    heatMapMenu.y = mouse.y
                    heatMapMenu.open()
                }
            }
        }
    }  // Qan.HeatMapPreview
}  // ApplicationWindow
