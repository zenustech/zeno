import QtQuick 2.12
import QtQuick.Controls 1.2
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import zeno.enum 1.0

Item {
    id: root
    property var graphModel: undefined
    property var graphView: undefined

    Repeater {
        model: root.graphModel ? root.graphModel.getLinkModel() : null
 
        delegate: Edge {
            id: current_edge
            required property var fromParam
            required property var toParam

            property int out_group: fromParam[2] ? ParamGroup.OutputObject : ParamGroup.OutputPrimitive
            property int in_group: toParam[2] ? ParamGroup.InputObject : ParamGroup.InputPrimitive

            visible: true
            point1x: 0
            point1y: 0
            point2x: 0
            point2y: 0
            p1_group: out_group
            in_key: toParam[3]
            //color: fromParam[2] ? "#7D2020" : "#4E9EF4"
            color: {
                if (toParam[3] != "") {
                    return Qt.rgba(1, 1, 1, 0.6)
                }
                else if (out_group == ParamGroup.OutputObject || in_group == ParamGroup.InputObject) {
                    return Qt.rgba(255/255, 255/255, 255/255, 0.6)
                }
                else {
                    return Qt.rgba(71/255, 165/255, 226/255, 1.0)
                }
            }
            color_hover: "#FFFFFF"
            color_selected: Qt.rgba(250/255, 100/255, 0, 1.0)
            z: 100

            thickness: {
                if (toParam[3] != "") {
                    return 1;
                }
                else if (out_group == ParamGroup.OutputObject || in_group == ParamGroup.InputObject) {
                    return 3;
                }
                else {
                    return 1;
                }
            }

            onStateChanged: function (state) {
                isSelected = (state == "select") ? true : false;
            }

            Component.onCompleted: {
                point1x = Qt.binding(function() {
                    var out_node_uuidpath = fromParam[0]
                    var out_nodeitem = root.graphView.graph.getNode(out_node_uuidpath).item

                    var out_sock_name = fromParam[1]
                    var out_sockitem = out_nodeitem.getSocketObject(out_sock_name, out_group)
                    if (out_sockitem == null) {
                        console.error("out_socketitem `" + out_sock_name + "` doesn't exist")
                        return 0
                    }

                    var outsock_pos_in_grid = null
                    if (out_sockitem.visible == false && out_group == ParamGroup.OutputPrimitive) {
                        outsock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(out_nodeitem.mapToGlobal(Qt.point(out_nodeitem.width, out_nodeitem.height/2)))
                    }
                    else {
                        outsock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(out_sockitem.mapToGlobal(Qt.point(out_sockitem.width/2, out_sockitem.height/2)))
                    }

                    out_nodeitem.x
                    out_nodeitem.y
                    out_nodeitem.width

                    if (out_nodeitem.node.group) {
                        out_nodeitem.node.group.item.x
                        out_nodeitem.node.group.item.y
                    }

                    return outsock_pos_in_grid.x
                })

                point1y = Qt.binding(function() {
                    var out_nodeitem = root.graphView.graph.getNode(fromParam[0]).item
                    var out_sockitem = out_nodeitem.getSocketObject(fromParam[1], out_group)
                    if (out_sockitem == null) {
                        console.error("out_socketitem `" + out_sock_name + "` doesn't exist")
                        return 0
                    }

                    var outsock_pos_in_grid = null
                    if (out_sockitem.visible == false && out_group == ParamGroup.OutputPrimitive) {
                        outsock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(out_nodeitem.mapToGlobal(Qt.point(out_nodeitem.width, out_nodeitem.height/2)))
                    }
                    else {
                        outsock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(out_sockitem.mapToGlobal(Qt.point(out_sockitem.width/2, out_sockitem.height/2)))
                    }

                    out_nodeitem.x
                    out_nodeitem.y
                    out_nodeitem.height

                    if (out_nodeitem.node.group) {
                        out_nodeitem.node.group.item.x
                        out_nodeitem.node.group.item.y
                    }

                    return outsock_pos_in_grid.y
                })

                point2x = Qt.binding(function() {
                    var in_nodeitem = root.graphView.graph.getNode(toParam[0]).item
                    var in_sockitem = in_nodeitem.getSocketObject(toParam[1], in_group)
                    if (in_sockitem == null) {
                        console.error("out_socketitem `" + toParam[1] + "` doesn't exist")
                        return 0
                    }

                    var insock_pos_in_grid = null
                    if (in_sockitem.visible == false) {
                        // console.log("in_so)
                        insock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(in_nodeitem.mapToGlobal(Qt.point(0, in_nodeitem.height/2)))
                    }
                    else{
                        insock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(in_sockitem.mapToGlobal(Qt.point(in_sockitem.width/2, in_sockitem.height/2)))
                    }
                    in_nodeitem.x
                    in_nodeitem.y
                    in_nodeitem.width
                    if (in_nodeitem.node.group) {
                        in_nodeitem.node.group.item.x
                        in_nodeitem.node.group.item.y
                    }
                    return insock_pos_in_grid.x
                })

                point2y = Qt.binding(function() {
                    var in_nodeitem = root.graphView.graph.getNode(toParam[0]).item
                    var in_sockitem = in_nodeitem.getSocketObject(toParam[1], in_group)
                    if (in_sockitem == null) {
                        console.error("in_sockitem `" + toParam[1] + "` doesn't exist")
                        return 0
                    }

                    var insock_pos_in_grid = null
                    if (in_sockitem.visible == false) {
                        insock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(in_nodeitem.mapToGlobal(Qt.point(0, in_nodeitem.height/2)))
                    }
                    else{
                        insock_pos_in_grid = root.graphView.containerItem.mapFromGlobal(in_sockitem.mapToGlobal(Qt.point(in_sockitem.width/2, in_sockitem.height/2)))
                    }
                    in_nodeitem.x
                    in_nodeitem.y
                    in_nodeitem.height

                    if (in_nodeitem.node.group) {
                        in_nodeitem.node.group.item.x
                        in_nodeitem.node.group.item.y
                    }

                    return insock_pos_in_grid.y
                })
            }
        }
    }

    Component.onCompleted: {
        
    }
}