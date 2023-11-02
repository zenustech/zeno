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

import QtQuick.Controls 2.15

import QtQuick.Layouts  1.3
import QtQuick.Controls.Material 2.1
import QtQuick.Shapes            1.0

import Qt.labs.platform 1.1 as Labs

import QuickQanava      2.0 as Qan
import TopologySample   1.0 as Qan
import "qrc:/QuickQanava" as Qan

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 720 // MPEG - 2 HD 720p - 1280 x 720 16:9
    title: "Topology test"
    Pane {
        anchors.fill: parent
        padding: 0
    }
    ScreenshotPopup {
        id: screenshotPopup
        graphView: graphView
    }
    Menu {
        id: menu
        title: "Main Menu"
        closePolicy: Popup.CloseOnPressOutside | Popup.CloseOnEscape
        property var targetNode: undefined
        property var targetGroup: undefined
        property var targetEdge: undefined
        onClosed: resetMenu()
        function resetMenu() {
            menu.targetNode = undefined
            menu.targetGroup = undefined
            menu.targetEdge = undefined
        }
        MenuItem {
            text: "Insert Node"
            onTriggered: {
                var n = topology.insertNode()
                centerItem(n.item)
                n.label = "Node #" + topology.getNodeCount()
                menu.resetMenu()
            }
        }
        MenuItem {
            text: {
                if (topology.selectedNodes.length > 1)
                    return "Remove All"
                else if (menu.targetGroup !== undefined)
                    return "Remove Group"
                return "Remove node"
            }
            enabled: menu.targetNode !== undefined ||
                     menu.targetGroup !== undefined ||
                     topology.selectedNodes.length > 1
            onTriggered: {
                if (topology.selectedNodes.length > 1) {
                    let nodes = []  // Copy the original selection, since removing nodes also modify selection
                    var n = 0
                    for (n = 0; n < topology.selectedNodes.length; n++)
                        nodes.push(topology.selectedNodes.at(n))
                    for (n = 0; n < nodes.length; n++) {
                        let node = nodes[n]
                        console.error('node.isGroup=' + node.isGroup())
                        topology.removeNode(nodes[n])
                    }
                } else if (menu.targetNode !== undefined)
                    topology.removeNode(menu.targetNode)
                else if (menu.targetGroup !== undefined)
                    topology.removeGroup(menu.targetGroup)
                menu.targetNode = undefined
            }
        }
        MenuItem {
            text: "Remove edge"
            enabled: menu.targetEdge !== undefined
            onTriggered: {
                if (menu.targetEdge !== undefined)
                    topology.removeEdge(menu.targetEdge)
                menu.targetEdge = undefined
            }
        }
        MenuItem {
            text: "Insert Group"
            onTriggered: {
                var n = topology.insertGroup()
                centerItem(n.item)
                n.label = "Group #" + topology.getGroupCount()
            }
        }
        Menu {
            title: "Align"
            MenuItem {
                text: "Align Horizontal Center"
                icon.name: 'ink-align-horizontal-center-symbolic'
                enabled: (topology.selectedNodes.length + topology.selectedGroups.length) > 1
                onTriggered: topology.alignSelectionHorizontalCenter()
            }
            MenuItem {
                text: "Align Left"
                icon.name: 'ink-align-horizontal-left-symbolic'
                enabled: (topology.selectedNodes.length + topology.selectedGroups.length) > 1
                onTriggered: topology.alignSelectionLeft()
            }
            MenuItem {
                text: "Align Right"
                icon.name: 'ink-align-horizontal-right-symbolic'
                enabled: (topology.selectedNodes.length + topology.selectedGroups.length) > 1
                onTriggered: topology.alignSelectionRight()
            }
            MenuItem {
                text: "Align Top"
                icon.name: 'ink-align-vertical-top-symbolic'
                enabled: (topology.selectedNodes.length + topology.selectedGroups.length) > 1
                onTriggered: topology.alignSelectionTop()
            }
            MenuItem {
                text: "Align Bottom"
                icon.name: 'ink-align-vertical-bottom-symbolic'
                enabled: (topology.selectedNodes.length + topology.selectedGroups.length) > 1
                onTriggered: topology.alignSelectionBottom()
            }
        } // Menu: align
        Menu {
            title: "Ports"
            MenuItem {
                text: "Add Left port"
                enabled: menu.targetNode !== undefined
                onTriggered: {
                    var inPort = topology.insertPort(menu.targetNode,
                                                     Qan.NodeItem.Left)
                    inPort.label = "LPORT"
                }
            }
            MenuItem {
                text: "Add Top port"
                enabled: menu.targetNode !== undefined
                onTriggered: topology.insertPort(menu.targetNode,
                                                 Qan.NodeItem.Top, "IN")
            }
            MenuItem {
                text: "Add Right port"
                enabled: menu.targetNode !== undefined
                onTriggered: topology.insertPort(menu.targetNode,
                                                 Qan.NodeItem.Right, "RPORT")
            }
            MenuItem {
                text: "Add Bottom port"
                enabled: menu.targetNode !== undefined
                onTriggered: topology.insertPort(menu.targetNode,
                                                 Qan.NodeItem.Bottom, "IN")
            }
        } // Menu: ports
        MenuSeparator { }
        MenuItem {
            text: "Show Radar"
            onTriggered: graphPreview.visible = checked
            checkable: true
            checked: graphPreview.visible
        }
        MenuItem {
            text: "Export to PNG"
            onTriggered: screenshotPopup.open()
        }
        MenuItem {
            text: "Fit Graph in View"
            onTriggered: graphView.fitContentInView()
        }
        MenuItem {
            text: "Clear Graph"
            onTriggered: topology.clearGraph()
        }
    } // Menu: menu

    Menu {
        id: menuRmPort
        title: "Port Menu"
        property var targetPort: undefined

        MenuItem {
            text: "Remove port"
            enabled: menuRmPort.targetPort !== undefined
            onTriggered: {
                if (menuRmPort.targetPort !== undefined)
                    topology.removePort(menuRmPort.targetPort.node,
                                        menuRmPort.targetPort)
                menuRmPort.targetPort = undefined
            }
        }
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

        Qan.FaceGraph {
            id: topology
            objectName: "graph"
            anchors.fill: parent
            clip: true
            connectorEnabled: true
            selectionColor: Material.accent
            connectorColor: Material.accent
            connectorEdgeColor: Material.accent
            onConnectorEdgeInserted: edge => {
                if (edge)
                    edge.label = "My edge"
            }
            property Component faceNodeComponent: Qt.createComponent("qrc:/FaceNode.qml")
            onNodeClicked: node => {
                portsListView.model = node.item.ports
            }
            onNodeRightClicked: (node, pos) => {
                const globalPos = node.item.mapToItem(topology, pos.x, pos.y)
                menu.x = globalPos.x
                menu.y = globalPos.y
                menu.targetNode = node
                menu.open()
            }
            onGroupRightClicked: (group, pos) => {
                const globalPos = group.item.mapToItem(topology, pos.x, pos.y)
                menu.x = globalPos.x
                menu.y = globalPos.y
                menu.targetGroup = group
                menu.open()
            }
            onEdgeRightClicked: (edge, pos) => {
                if (!edge || !edge.item)
                    return
                const globalPos = edge.item.mapToItem(topology, pos.x, pos.y)
                menu.x = globalPos.x
                menu.y = globalPos.y
                menu.targetEdge = edge
                menu.open()
            }
            onPortRightClicked: (port, pos) => {
                var globalPos = port.parent.mapToItem(topology, pos.x, pos.y)
                menuRmPort.x = globalPos.x
                menuRmPort.y = globalPos.y
                menuRmPort.targetPort = port
                menuRmPort.open()
            }

            Component.onCompleted: {
                defaultEdgeStyle.lineWidth = 3
                defaultEdgeStyle.lineColor = Qt.binding(function () {
                    return Material.foreground
                })
                defaultNodeStyle.shadowColor = Qt.binding(function () {
                    return Material.theme === Material.Dark ? Qt.darker(
                                                                  Material.foreground) : Qt.darker(
                                                                  Material.foreground)
                })
                defaultNodeStyle.backColor = Qt.binding(function () {
                    return Material.theme === Material.Dark ? Qt.lighter(
                                                                  Material.background) : Qt.lighter(
                                                                  Material.background)
                })
                defaultGroupStyle.backColor = Qt.binding(function () {
                    return Material.theme === Material.Dark ? Qt.lighter(
                                                                  Material.background,
                                                                  1.3) : Qt.darker(
                                                                  Material.background,
                                                                  1.1)
                })
                var bw1 = topology.insertFaceNode()
                bw1.image = "qrc:/faces/BW1.jpg"
                bw1.item.x = 150
                bw1.item.y = 55
                var bw1p1 = topology.insertPort(bw1, Qan.NodeItem.Right)
                bw1p1.label = "P#1"
                var bw1p2 = topology.insertPort(bw1, Qan.NodeItem.Bottom)
                bw1p2.label = "P#2"
                var bw1p3 = topology.insertPort(bw1, Qan.NodeItem.Bottom)
                bw1p3.label = "P#3"

                var bw2 = topology.insertFaceNode()
                bw2.image = "qrc:/faces/BW2.jpg"
                bw2.item.x = 45
                bw2.item.y = 250
                var bw2p1 = topology.insertPort(bw2, Qan.NodeItem.Top)
                bw2p1.label = "P#1"

                var bw3 = topology.insertFaceNode()
                bw3.image = "qrc:/faces/BW3.jpg"
                bw3.item.x = 250
                bw3.item.y = 250
                var bw3p1 = topology.insertPort(bw3, Qan.NodeItem.Top)
                bw3p1.label = "P#1"

                var js1 = topology.insertFaceNode()
                js1.image = "qrc:/faces/JS1.jpg"
                js1.item.x = 500
                js1.item.y = 55
                var js1p1 = topology.insertPort(js1, Qan.NodeItem.Left)
                js1p1.label = "P#1"
                var js1p2 = topology.insertPort(js1, Qan.NodeItem.Bottom)
                js1p2.label = "P#2"
                var js1p3 = topology.insertPort(js1, Qan.NodeItem.Bottom)
                js1p3.label = "P#3"
                var js1p4 = topology.insertPort(js1, Qan.NodeItem.Top)
                js1p4.label = "P#4"

                var js2 = topology.insertFaceNode()
                js2.image = "qrc:/faces/JS2.jpg"
                js2.item.x = 500
                js2.item.y = -155
                var js2p1 = topology.insertPort(js2, Qan.NodeItem.Bottom)
                js2p1.label = "P#1"

                var vd1 = topology.insertFaceNode()
                vd1.image = "qrc:/faces/VD1.jpg"
                vd1.item.x = 400
                vd1.item.y = 350
                var vd1p1 = topology.insertPort(vd1, Qan.NodeItem.Top)
                vd1p1.label = "P#1"
                var vd1p2 = topology.insertPort(vd1, Qan.NodeItem.Bottom)
                vd1p2.label = "P#2"
                var vd1p3 = topology.insertPort(vd1, Qan.NodeItem.Bottom)
                vd1p3.label = "P#3"

                var vd2 = topology.insertFaceNode()
                vd2.image = "qrc:/faces/VD2.jpg"
                vd2.item.x = 200
                vd2.item.y = 600
                var vd2p1 = topology.insertPort(vd2, Qan.NodeItem.Top)
                vd2p1.label = "P#1"

                var vd3 = topology.insertFaceNode()
                vd3.image = "qrc:/faces/VD3.jpg"
                vd3.item.x = 400
                vd3.item.y = 600
                var vd3p1 = topology.insertPort(vd3, Qan.NodeItem.Top)
                vd3p1.label = "P#1"

                var dd1 = topology.insertFaceNode()
                dd1.image = "qrc:/faces/DD1.jpg"
                dd1.item.x = 650
                dd1.item.y = 350
                var dd1p1 = topology.insertPort(dd1, Qan.NodeItem.Top)
                dd1p1.label = "P#1"
                var dd1p2 = topology.insertPort(dd1, Qan.NodeItem.Bottom)
                dd1p2.label = "P#2"
                var dd1p3 = topology.insertPort(dd1, Qan.NodeItem.Bottom)
                dd1p3.label = "P#3"

                var dd2 = topology.insertFaceNode()
                dd2.image = "qrc:/faces/DD2.jpg"
                dd2.item.x = 650
                dd2.item.y = 600
                var dd2p1 = topology.insertPort(dd2, Qan.NodeItem.Top)
                dd2p1.label = "P#1"

                var dd3 = topology.insertFaceNode()
                dd3.image = "qrc:/faces/DD3.jpg"
                dd3.item.x = 800
                dd3.item.y = 600
                var dd3p1 = topology.insertPort(dd3, Qan.NodeItem.Top)
                dd3p1.label = "P#1"

                /* e = topology.insertEdge(bw2, bw1)
                 topology.bindEdgeSource(e, bw2p1)
                 topology.bindEdgeDestination(e, bw1p2)
                 e = topology.insertEdge(bw3, bw1)
                 topology.bindEdgeSource(e, bw3p1)
                 topology.bindEdgeDestination(e, bw1p3)

                 e = topology.insertEdge(js1, js2)
                 topology.bindEdgeSource(e, js1p4)
                 topology.bindEdgeDestination(e, js2p1)

                 e = topology.insertEdge(js1, vd1)
                 topology.bindEdgeSource(e, js1p2)
                 topology.bindEdgeDestination(e, vd1p1)

                 e = topology.insertEdge(js1, dd1)
                 topology.bindEdgeSource(e, js1p3)
                 topology.bindEdgeDestination(e, dd1p1)

                 e = topology.insertEdge(dd2, dd1)
                 topology.bindEdgeSource(e, dd2p1)
                 topology.bindEdgeDestination(e, dd1p2)

                 e = topology.insertEdge(dd3, dd1)
                 topology.bindEdgeSource(e, dd3p1)
                 topology.bindEdgeDestination(e, dd1p3)

                 e = topology.insertEdge(vd2, vd1)
                 topology.bindEdgeSource(e, vd2p1)
                 topology.bindEdgeDestination(e, vd1p2)

                 e = topology.insertEdge(vd3, vd1)
                 topology.bindEdgeSource(e, vd3p1)
                 topology.bindEdgeDestination(e, vd1p3)*/
            }
        } // Qan.Graph: graph
        onRightClicked: pos => {
            //let globalPos = graphView.mapToItem(topology, pos.x, pos.y)
            menu.targetNode = undefined
            menu.targetEdge = undefined
            menu.open()
        }
    }
    Label {
        text: "Right click for main menu:
\t- Add content with Add Node or Add Face Node entries.
\t- Use the DnD connector to add edges between nodes."
    }

    RowLayout {
        id: topDebugLayout
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.rightMargin: 15
        anchors.topMargin: 5
        spacing: 15
        Frame {
            id: edgesListView
            Layout.preferredWidth: 200
            Layout.preferredHeight: 200
            Layout.alignment: Qt.AlignTop
            visible: showDebugControls.checked
            leftPadding: 0; rightPadding: 0
            topPadding: 0;  bottomPadding: 0
            Pane { anchors.fill: parent; anchors.margins: 1; opacity: 0.7 }
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label {
                    Layout.margins: 3
                    text: "Edges:"
                    font.bold: true
                }
                EdgesListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: topology.edges
                }
            }
        } // Frame: edgesListView

        Frame {
            Layout.preferredWidth: 200
            Layout.preferredHeight: 300
            leftPadding: 0; rightPadding: 0
            topPadding: 0;  bottomPadding: 0
            visible: showDebugControls.checked
            padding: 0
            Pane { anchors.fill: parent; anchors.margins: 1; opacity: 0.7 }
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label {
                    Layout.margins: 3
                    text: "Nodes:"
                    font.bold: true
                }
                NodesListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: topology.nodes
                    graphView: graphView
                }
            }
        } // Frame: nodesListView

        Frame {
            id: portList
            Layout.preferredWidth: 200
            Layout.preferredHeight: 300
            visible: showDebugControls.checked
            leftPadding: 0; rightPadding: 0
            topPadding: 0;  bottomPadding: 0
            Pane { anchors.fill: parent; anchors.margins: 1; opacity: 0.7 }
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label {
                    Layout.margins: 3
                    text: "Selected Node's Ports:"
                    font.bold: true
                }
                ListView {
                    id: portsListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 4
                    focus: true
                    flickableDirection: Flickable.VerticalFlick
                    highlightFollowsCurrentItem: false
                    highlight: Rectangle {
                        x: 0
                        y: portsListView.currentItem ? portsListView.currentItem.y : 0
                        width: portsListView.width
                        height: portsListView.currentItem ? portsListView.currentItem.height : 0
                        color: Material.accent
                        opacity: 0.7
                        radius: 3
                        visible: portsListView.currentItem !== undefined && portsListView.currentItem !== null
                        Behavior on y {
                            SpringAnimation {
                                duration: 200
                                spring: 2
                                damping: 0.1
                            }
                        }
                    }
                    delegate: Item {
                        id: portDelegate
                        width: ListView.view.width
                        height: 30
                        Label {
                            id: portLabel
                            text: "Label: " + itemData.label
                        }
                        MouseArea {
                            anchors.fill: portDelegate
                            onClicked: {
                                portsListView.currentIndex = index
                            }
                        }
                    }
                }
            }
        } // portList
    }  // RowLayout nodes / nodes ports / edge debug control

    Labs.ColorDialog {
        id: selectionColorDialog
        title: "Selection hilight color"
        onAccepted: {
            topology.selectionColor = color
        }
    }

    Frame {
        id: selectionView

        anchors.top: topDebugLayout.bottom
        anchors.topMargin: 15
        anchors.right: parent.right
        anchors.rightMargin: 15
        width: 250
        height: 280

        visible: showDebugControls.checked

        leftPadding: 0; rightPadding: 0
        topPadding: 0;  bottomPadding: 0

        Pane { anchors.fill: parent; anchors.margins: 1; opacity: 0.7 }
        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            Label {
                Layout.margins: 3
                text: "Selection:"
                font.bold: true
                horizontalAlignment: Text.AlignLeft
            }
            ListView {
                id: selectionListView
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                model: topology.selectedNodes
                spacing: 4
                focus: true
                flickableDirection: Flickable.VerticalFlick
                highlightFollowsCurrentItem: false
                highlight: Rectangle {
                    x: 0
                    y: (selectionListView.currentItem
                        !== null ? selectionListView.currentItem.y : 0)
                    width: selectionListView.width
                    height: selectionListView.currentItem ? selectionListView.currentItem.height : 0
                    color: Material.accent
                    opacity: 0.7
                    radius: 3
                    Behavior on y {
                        SpringAnimation {
                            duration: 200
                            spring: 2
                            damping: 0.1
                        }
                    }
                }
                delegate: Item {
                    id: selectedNodeDelegate
                    width: ListView.view.width
                    height: 30
                    Label { text: "Label: " + itemData.label }
                    MouseArea {
                        anchors.fill: selectedNodeDelegate
                        onClicked: {
                            selectedNodeDelegate.ListView.view.currentIndex = index
                        }
                    }
                }
            }
            RowLayout {
                Layout.margins: 2
                Label { text: "Policy:" }
                Item { Layout.fillWidth: true } // Space eater
                ColumnLayout {
                    CheckBox {
                        Layout.preferredHeight: 25
                        height: 15
                        autoExclusive: true
                        text: "NoSelection"
                        checked: topology.selectionPolicy === Qan.Graph.NoSelection
                        onCheckedChanged: {
                            if (checked)
                                topology.selectionPolicy = Qan.Graph.NoSelection
                        }
                    }
                    CheckBox {
                        Layout.preferredHeight: 25
                        height: 15
                        autoExclusive: true
                        text: "SelectOnClick"
                        checked: topology.selectionPolicy === Qan.Graph.SelectOnClick
                        onCheckedChanged: {
                            if (checked)
                                topology.selectionPolicy = Qan.Graph.SelectOnClick
                        }
                    }
                    CheckBox {
                        Layout.preferredHeight: 25
                        height: 15
                        autoExclusive: true
                        text: "SelectOnCtrlClick"
                        checked: topology.selectionPolicy === Qan.Graph.SelectOnCtrlClick
                        onCheckedChanged: {
                            if (checked)
                                topology.selectionPolicy = Qan.Graph.SelectOnCtrlClick
                        }
                    }
                }
            }
            RowLayout {
                Layout.margins: 2
                Label { text: "Color:" }
                Item { Layout.fillWidth: true }        // Space eater
                Rectangle {
                    Layout.preferredWidth: 25
                    Layout.preferredHeight: 25
                    color: topology.selectionColor
                    radius: 3
                    border.width: 1
                    border.color: Qt.lighter(topology.selectionColor)
                }
                Button {
                    Layout.preferredHeight: 30
                    Layout.preferredWidth: 30
                    text: "..."
                    onClicked: {
                        selectionColorDialog.color = topology.selectionColor
                        selectionColorDialog.open()
                    }
                }
            }
            RowLayout {
                Layout.margins: 2
                Label { text: "Weight:" }
                Slider {
                    Layout.preferredHeight: 20
                    Layout.fillWidth: true
                    from: 1.0
                    to: 15.
                    stepSize: 0.1
                    value: topology.selectionWeight
                    onValueChanged: {
                        topology.selectionWeight = value
                    }
                }
            }
            RowLayout {
                Layout.margins: 2
                Label {
                    text: "Margin:"
                }
                Slider {
                    Layout.preferredHeight: 20
                    Layout.fillWidth: true
                    from: 1.0
                    to: 15.
                    stepSize: 0.1
                    value: topology.selectionMargin
                    onValueChanged: {
                        topology.selectionMargin = value
                    }
                }
            }
        }
    } // selectionView


    Control {
        anchors.bottom: parent.bottom
        anchors.left: parent.left

        width: 470
        height: 50
        padding: 0

        Pane {
            anchors.fill: parent
            opacity: 0.55
        }
        RowLayout {
            anchors.fill: parent
            anchors.margins: 0
            CheckBox {
                text: qsTr("Dark")
                checked: ApplicationWindow.contentItem.Material.theme === Material.Dark
                onClicked: ApplicationWindow.contentItem.Material.theme
                           = checked ? Material.Dark : Material.Light
            }
            RowLayout {
                Layout.margins: 2
                Label {
                    text: "Edge type:"
                }
                Item {
                    Layout.fillWidth: true
                }
                ComboBox {
                    model: ["Straight", "Curved"]
                    enabled: defaultEdgeStyle !== undefined
                    currentIndex: defaultEdgeStyle.lineType === Qan.EdgeStyle.Straight ? 0 : 1
                    onActivated: {
                        if (index == 0)
                            defaultEdgeStyle.lineType = Qan.EdgeStyle.Straight
                        else if (index == 1)
                            defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
                    }
                }
                CheckBox {
                    id: showDebugControls
                    text: "Show Debug controls"
                    checked: false
                }
            } // RowLayout: edgeType
        }
    }

    Qan.GraphPreview {
        id: graphPreview
        source: graphView
        viewWindowColor: Material.accent
        anchors.right: graphView.right; anchors.bottom: graphView.bottom
        anchors.rightMargin: 8; anchors.bottomMargin: 8
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
}  // ApplicationWindow
