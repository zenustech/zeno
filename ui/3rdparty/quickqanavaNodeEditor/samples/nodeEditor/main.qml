import QtQuick                   2.3
import QtQuick.Controls          2.3
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
//import Qt.labs.platform          1.1  // ColorDialog

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import "." as Qan

ApplicationWindow {
    id: window
    visible: true
    width: 1280; height: 720
    title: "Edge/Visual connector sample"

    Pane { anchors.fill: parent }

    Qan.GraphView {
        id: graphView
        anchors.fill: parent
        navigable   : true
        /*PinchHandler {
            target: null
            onActiveScaleChanged: {
                console.error('centroid.position=' + centroid.position)
                console.error('activeScale=' + activeScale)
                var p = centroid.position
                var f = activeScale > 1.0 ? 1. : -1.
                graphView.zoomOn(p, graphView.zoom + (f * 0.04))
            }
        }*/
        graph: Qan.Graph {
            parent: graphView
            id: topology
            anchors.fill: parent
            objectName: "graph"
            clip: true
            connectorEnabled: true
            property int portHeight: 16

            nodeDelegate: MyNode{
                portHeight: topology.portHeight
            }
        
            portDelegate: Component {
                id: youpi
                Qan.PortItem {
                    id: portit
                    y:20
                    width: 16; height: topology.portHeight
                    property alias txt:lbl.text 
                    property alias lb:lbl  
                    property alias por:port 
                    Label {
                        id:lbl
                            //Layout.preferredWidth: height
                            text: "position"
                            color: "#FFFFFF"
                            font{bold:true; pixelSize: 15}
                            //rotation: -90
                        }
                    Rectangle {
                        id:port
                        radius: 8
                        anchors.fill: parent
                        color: "#3B6491"
                        border.color: "#4B9EF4"; border.width: 3
                    }
                    states: [
                        State {
                            name: "left"
                            when: portit !== null && portit !== undefined && portit.dockType === Qan.NodeItem.Left
                            AnchorChanges {
                                target: lbl
                                anchors {
                                    left: port.right
                                }
                            }
                        },
                        State {
                            name: "right"
                            when: portit !== null && portit !== undefined && portit.dockType === Qan.NodeItem.Right
                            AnchorChanges {
                                target: lbl
                                anchors {
                                    right: port.left
                                }
                            }
                        }
                    ]
                }
            }

            Component.onCompleted: {
                defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved

                var nodeData = {
                    name:"DefaultNode1",
                    x:30,
                    y:30,
                    //var componentLst = ["text", "textedit", "slider", "checkbox", "combobox", "fileinput"]

                    //widgetLst : ["text","text"],
                    //portNameLst : ["p1","p7"],
                    //portTypeLst : ["in", "out"]

                    widgetLst : ["text", "textedit", "slider", "checkbox", "fileinput", "vec2text", "vec3text", "vec4text"],
                    portNameLst : ["p1", "p2","p3","p4","p5","p6","p7","p8"],
                    portTypeLst : ["in","in","in","in","in","in","out","out"]
                }

                for(var j = 0; j < 1; j++){
                    for(var i = 0; i < 2; i++){
                        nodeData.x = 1200 * j
                        nodeData.y = i*500
                        nodeData.name = "DefaultNode" + i*2
                        var n1 = topology.insertNode(nodeDelegate)
                        n1.item.initializeNode(topology, n1, nodeData)
                        var p7 = n1.item.findPort(nodeData.name + "p7")
                        //var p8 = n1.item.findPort(nodeData.name + "p8")

                        nodeData.x = 1200 * j + 600
                        nodeData.y = i*500
                        nodeData.name = "DefaultNode" + (i*2 + 1)
                        var n2 = topology.insertNode(nodeDelegate)
                        n2.item.initializeNode(topology, n2, nodeData)
                        var p1 = n2.item.findPort(nodeData.name + "p1")
                        //var p2 = n2.item.findPort(nodeData.name + "p2")

                        var e = topology.insertEdge(n1, n2);
                        e.item.dstShape = Qan.EdgeStyle.None
                        topology.bindEdgeSource(e, p7)
                        topology.bindEdgeDestination(e, p1)

                        //var e2 = topology.insertEdge(n1, n2);
                        //topology.bindEdgeSource(e2, p8)
                        //topology.bindEdgeDestination(e2, p2)
                    }
                }
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
        }
    }
    Qan.GraphPreview {
        id: graphPreview
        source: graphView
        viewWindowColor: Material.accent
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: graphView.bottom
        anchors.bottomMargin: 8
        width: 350
        height: 198
    }  // Qan.GraphPreview

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
            text: "Remove edge"
            enabled: menu.targetEdge !== undefined
            onTriggered: {
                if (menu.targetEdge !== undefined)
                    topology.removeEdge(menu.targetEdge)
                menu.targetEdge = undefined
            }
        }
    }
}