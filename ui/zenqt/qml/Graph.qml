import QtQuick 2.3
import QtQuick.Controls 2.3
import QtQml.Models 2.1

Item {
    id: graphEditor
    anchors.fill: parent
    property real maxZoom: 2.0
    property real minZoom: 0.1
    property variant graphModel

    MouseArea {
        id: graphEditorArea
        anchors.fill: parent
        property double factor: 1.15
        // Activate multisampling for edges antialiasing
        layer.enabled: true
        layer.samples: 8

        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
        drag.threshold: 0
        onWheel: {
            var zoomFactor = wheel.angleDelta.y > 0 ? factor : 1/factor
            var scale = draggable.scale * zoomFactor
            scale = Math.min(Math.max(minZoom, scale), maxZoom)
            if(draggable.scale == scale)
                return
            var point = mapToItem(draggable, wheel.x, wheel.y)
            draggable.x += (1-zoomFactor) * point.x * draggable.scale
            draggable.y += (1-zoomFactor) * point.y * draggable.scale
            draggable.scale = scale
        }
        onPressed: {
            if (mouse.button == Qt.MiddleButton) {
                drag.target = draggable // start drag
            }
        }
        onReleased: {
            drag.target = undefined // stop drag
        }

        onClicked:{
            tempEdge.visible = false
        }

        Item {
            id: draggable

            Repeater{
                id: nodes
                model: graphEditor.graphModel

                function idxFromId(ident) {
                    var idx = graphEditor.graphModel.indexFromId(ident)
                    //console.log(idx)
                }

                function getZNode(ident) {
                    var idx = graphEditor.graphModel.indexFromId(ident)
                    if (idx == -1) {
                        return null
                    } else {
                        return nodes.itemAt(idx)
                    }
                }

                delegate: ZNode {
                    required property string name
                    required property string ident
                    required property variant params
                    required property var pos
                    required property variant subgraph

                    //id: ident     //warning: Unable to assign ZNode_QMLTYPE_31_QML_35 to QString
                    id: qmlnode

                    arg_name: name
                    arg_ident: ident
                    paramModel: params
                    subgModel: subgraph
                    x: pos[0]
                    y: pos[1]

                    addLink: (sockObj) => {
                        if (tempEdge.visible && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident){
                            tempEdge.visible = false
                            tempEdge.isMatch = false
                            if (!tempEdge.isFromInput){
                                graphEditor.graphModel.addLink(tempEdge.nodeId, tempEdge.paramName, ident, sockObj.paramName)
                            }
                            else {
                                graphEditor.graphModel.addLink(ident, sockObj.paramName, tempEdge.nodeId, tempEdge.paramName)
                            }
                        }
                    }

                    getTempEdge: () => {
                        return tempEdge
                    }

                    matchSocket: (sockObj) => {//吸附
                        if (tempEdge.visible && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident) {
                            var sockGlobalPos = draggable.mapFromItem(sockObj, 0, 0)
                            if (sockObj.input){
                                tempEdge.point2x = sockGlobalPos.x
                                tempEdge.point2y = sockGlobalPos.y + sockObj.height/2
                            }
                            else {
                                tempEdge.point1x = sockGlobalPos.x
                                tempEdge.point1y = sockGlobalPos.y + sockObj.height/2
                            }
                            tempEdge.isMatch = true
                        }
                    }

                    mismatchSocket: () => {//退出吸附
                        if (tempEdge.visible) {
                            
                            if (!tempEdge.isFromInput){
                                tempEdge.point2x = Qt.binding(function() {
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.x
                                })
                                tempEdge.point2y = Qt.binding(function() {
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.y
                                })
                            }
                            else {
                                tempEdge.point1x = Qt.binding(function() {
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.x
                                })
                                tempEdge.point1y = Qt.binding(function() {
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.y
                                })
                            }
                            tempEdge.isMatch = false
                            
                        }
                    }

                    sockOnClicked: (sockObj) => {
                        var sockGlobalPos = draggable.mapFromItem(sockObj, 0, 0)
                        console.log('sockGlobalPos: ' + sockGlobalPos.x + ',' + sockGlobalPos.y)

                        //点击将临时边连接变成固定边
                        if (tempEdge.isMatch && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident){
                           qmlnode.addLink(sockObj)
                        }
                        else if (sockObj.input) {
                            var fromParam = graphEditor.graphModel.removeLink(ident, sockObj.paramName, true)
                            if (fromParam != undefined && fromParam.length > 0){//删除边并变成临时边
                                tempEdge.visible = true
                                tempEdge.nodeId = fromParam[0]
                                tempEdge.isFromInput = false
                                tempEdge.paramName = fromParam[1]
                                tempEdge.point1x = Qt.binding(function() {
                                        var outNode = nodes.getZNode(fromParam[0])
                                        var outSocketObj = outNode.getSocketObj(fromParam[1], false)    
                                        var pt = outNode.mapFromItem(outSocketObj, 0, 0)
                                        return pt.x + outNode.x
                                    })

                                tempEdge.point1y = Qt.binding(function() {
                                    var outNode = nodes.getZNode(fromParam[0])
                                    var outSocketObj = outNode.getSocketObj(fromParam[1], false)  
                                    var pt = outNode.mapFromItem(outSocketObj, 0, 0)
                                    return pt.y + outNode.y + outSocketObj.height/2
                                })

                            
                                tempEdge.point2x = Qt.binding(function() {
                                    graphEditorArea.mouseX
                                     var mousePos = draggable.mapFromItem(draggable, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                     return mousePos.x
                                })
                                tempEdge.point2y = Qt.binding(function() {
                                    graphEditorArea.mouseY
                                     var mousePos = draggable.mapFromItem(draggable, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                     return mousePos.y
                                })
                                tempEdge.isMatch = false
                            }
                            else{//从 input 到 output 的临时边
                                tempEdge.visible = true
                                tempEdge.nodeId = ident
                                tempEdge.isFromInput = true
                                tempEdge.paramName = sockObj.paramName
                                tempEdge.point1x = Qt.binding(function() {
                                    //graphEditorArea.mouseX
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.x
                                })
                                tempEdge.point1y = Qt.binding(function() {
                                    //graphEditorArea.mouseY
                                    var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                    return mousePos.y
                                }) 
                                tempEdge.point2x = sockGlobalPos.x
                                tempEdge.point2y = sockGlobalPos.y + sockObj.height/2
                                tempEdge.isMatch = false
                            }
                        }
                        else {//从output 到input的临时边
                            tempEdge.visible = true
                            tempEdge.nodeId = ident
                            tempEdge.isFromInput = false
                            tempEdge.paramName = sockObj.paramName
                            tempEdge.point1x = sockGlobalPos.x
                            tempEdge.point1y = sockGlobalPos.y + sockObj.height/2
                            tempEdge.point2x = Qt.binding(function() {
                                //console.log('graphEditorArea.mouseX ' + graphEditorArea.mouseX)
                                //graphEditorArea.mouseX
                                var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                //console.log('mousePos.x: ' + mousePos.x)
                                return mousePos.x
                            })
                            tempEdge.point2y = Qt.binding(function() {
                                //graphEditorArea.mouseY
                                var mousePos = draggable.mapFromItem(graphEditorArea, graphEditorArea.mouseX, graphEditorArea.mouseY)
                                //console.log('mousePos.y: ' + mousePos.y)
                                return mousePos.y
                            })
                            tempEdge.isMatch = false
                        }
                    }

                    destoryTempEdge: () => {
                        tempEdge.visible = false
                    }

                }

                Component.onCompleted: {
                    var edgesContainer = Qt.createQmlObject('
                        import QtQuick 2.12
                        import QtQuick.Controls 1.2
                        import QtQuick.Layouts 1.3
                        import QtQuick.Controls 1.4
                        import QtQuick.Controls.Styles 1.4

                        Repeater {
                            model: graphEditor.graphModel.getLinkModel()

                            delegate: Edge {

                                required property var fromParam
                                required property var toParam

                                id: edge233
                                visible: true
                                point1x: 0
                                point1y: 0
                                point2x: 0
                                point2y: 0
                                color: "#4E9EF4"

                                Component.onCompleted: {
                                    point1x = Qt.binding(function() {
                                        var outNode = nodes.getZNode(fromParam[0])
                                        outNode.width       //��outNode�Ŀ��ȷ����仯ʱ,ǿ���䴥������,����ֻ�е�outNode.x�ƶ��Ż����, ����: ��������ʱ

                                        var socketObj = outNode.getSocketObj(fromParam[1], false)
                                        var pt = outNode.mapFromItem(socketObj, 0, 0)
                                        //console.log("x=", pt2.x)
                                        return pt.x + outNode.x
                                    })

                                    point1y = Qt.binding(function() {
                                        var outNode = nodes.getZNode(fromParam[0])
                                        outNode.height

                                        var socketObj = outNode.getSocketObj(fromParam[1], false)
                                        var pt = outNode.mapFromItem(socketObj, 0, 0)
                                        //console.log("y=", pt.y)
                                        return pt.y+ socketObj.height/2 + outNode.y
                                    })

                                    point2x = Qt.binding(function() {
                                        var inNode = nodes.getZNode(toParam[0])
                                        inNode.width

                                        var socketObj = inNode.getSocketObj(toParam[1], true)
                                        var pt = inNode.mapFromItem(socketObj, 0, 0)
                                        return inNode.x + pt.x
                                    })

                                    point2y = Qt.binding(function() {
                                        var inNode = nodes.getZNode(toParam[0])
                                        inNode.height

                                        var socketObj = inNode.getSocketObj(toParam[1], true)
                                        var pt = inNode.mapFromItem(socketObj, 0, 0)
                                        return inNode.y + pt.y +  socketObj.height/2
                                    })
                                }
                            }
                        }', draggable)
                }
            }

            Edge {
                id: tempEdge
                visible: false
                point1x: 0
                point1y: 0
                point2x: 0
                point2y: 0
                color: "#5FD2FF"
                thickness: 4
                isFromInput: false
            }
        } // EndItem
    }

    Menu {
        id: menu
        title: "Node Menu"
        closePolicy: Popup.CloseOnPressOutside | Popup.CloseOnEscape
        property var targetNodeRepeaterIndex: undefined
        onClosed: menu.targetNodeRepeaterIndex = undefined
        MenuItem {
            text: "Remove node"
            enabled: menu.targetNodeRepeaterIndex !== undefined
            onTriggered: {
                if (menu.targetNodeRepeaterIndex !== undefined)
                    myModel.remove(menu.targetNodeRepeaterIndex)
            }
        }
    }
}
