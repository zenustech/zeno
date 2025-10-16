import QtQuick          2.12
import QtQuick.Controls 2.0
import QtQuick.Layouts  1.3
import QtQuick.Shapes   1.0

import QuickQanava          2.0 as Qan
//import "qrc:/QuickQanava"   as Qan

Item {
    
        LineGrid { id: lineGrid }

        Qan.Navigable {
            id: navigable
            anchors.fill: parent
            clip: true
            navigable: true
            grid: lineGrid
            PinchHandler {
                target: null
                onActiveScaleChanged: {
                    console.error('centroid.position=' + centroid.position)
                    console.error('activeScale=' + activeScale)
                    var p = centroid.position
                    var f = activeScale > 1.0 ? 1. : -1.
                    navigable.zoomOn(p, navigable.zoom + (f * 0.03))
                }
            }
        }

    
        MouseArea {
            id: graphEditorArea
            parent: navigable.containerItem
            //parent: navigable
            anchors.fill: parent
            property double factor: 1.15
            // Activate multisampling for edges antialiasing
            layer.enabled: true
            layer.samples: 8
            //preventStealing: true

            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.RightButton | Qt.MiddleButton
            drag.threshold: 0

            onPressed: {
                console.log("<-----onPressed graphEditorArea------->")
                if (mouse.button == Qt.MiddleButton) {
                    drag.target = nodes // start drag
                }
            }

            onReleased: {
                console.log("<-----onReleased graphEditorArea------->")
                drag.target = undefined // stop drag
            
            }

            onClicked:{
                console.log("<-----clicked graphEditorArea------->")
                tempEdge.visible = false
            }

       }

        Item {
            id: draggable
            //parent: graphEditorArea
            anchors.fill: parent
            parent: navigable.containerItem
            Repeater{
                id: nodes
                model: nodesModel

                function idxFromId(ident) {
                    var idx = nodesModel.indexFromId(ident)
                    //console.log(idx)
                }

                function getZNode(ident) {
                    var idx = nodesModel.indexFromId(ident)
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

                    //id: ident     //warning: Unable to assign ZNode_QMLTYPE_31_QML_35 to QString
                    id: qmlnode

                    arg_name: name
                    arg_ident: ident
                    paramModel: params
                    x: pos[0]
                    y: pos[1]

                    addLink: (sockObj) => {
                        if (tempEdge.visible && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident){
                            tempEdge.visible = false
                            tempEdge.isMatch = false
                            if (!tempEdge.isFromInput){
                                nodesModel.addLink(tempEdge.nodeId, tempEdge.paramName, ident, sockObj.paramName)
                            }
                            else {
                                nodesModel.addLink(ident, sockObj.paramName, tempEdge.nodeId, tempEdge.paramName)
                            }
                        }
                    }

                    getTempEdge: () => {
                        return tempEdge
                    }

                    matchSocket: (sockObj) => {//吸附
                        if (tempEdge.visible && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident) {
                            var sockGlobalPos = graphEditorArea.mapFromItem(sockObj, 0, 0)
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
                                tempEdge.point2x = Qt.binding(function() { return graphEditorArea.mouseX })
                                tempEdge.point2y = Qt.binding(function() { return graphEditorArea.mouseY })
                            }
                            else {
                                tempEdge.point1x = Qt.binding(function() { return graphEditorArea.mouseX })
                                tempEdge.point1y = Qt.binding(function() { return graphEditorArea.mouseY })
                            }
                            tempEdge.isMatch = false
                        }
                    }

                    sockOnClicked: (sockObj) => {
                        var sockGlobalPos = graphEditorArea.mapFromItem(sockObj, 0, 0)
                        //点击将临时边连接变成固定边
                        if (tempEdge.isMatch && tempEdge.isFromInput != sockObj.input && tempEdge.nodeId != ident){
                            qmlnode.addLink(sockObj)
                        }
                        else if (sockObj.input) {
                            var fromParam = nodesModel.removeLink(ident, sockObj.paramName, true)
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
                                tempEdge.point2x = Qt.binding(function() { return graphEditorArea.mouseX })
                                tempEdge.point2y = Qt.binding(function() { return graphEditorArea.mouseY }) 
                                tempEdge.isMatch = false
                            }
                            else{//从 input 到 output 的临时边
                                tempEdge.visible = true
                                tempEdge.nodeId = ident
                                tempEdge.isFromInput = true
                                tempEdge.paramName = sockObj.paramName
                                tempEdge.point1x = Qt.binding(function() { return graphEditorArea.mouseX })
                                tempEdge.point1y = Qt.binding(function() { return graphEditorArea.mouseY }) 
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
                            tempEdge.point2x = Qt.binding(function() { return graphEditorArea.mouseX })
                            tempEdge.point2y = Qt.binding(function() { return graphEditorArea.mouseY })
                            tempEdge.isMatch = false
                            console.log("output ==> input: x = " + graphEditorArea.mouseX + "  + y = " + graphEditorArea.mouseY )
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
                            model: nodesModel.getLinkModel()

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
                        }', graphEditorArea)
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

            /*MouseArea {
                id: itemArea
                anchors.fill: parent
                onPressed: {
                    console.log("<-----onPressed itemArea------->")
                }
                onReleased: {
                    console.log("<-----onReleased itemArea------->")
                }

                onClicked:{
                    console.log("<-----clicked itemArea------->")
                }
            }*/
        }// End Item draggle
        

        /*
        Rectangle {
            parent: navigable.containerItem
            x: 100; y: 100
            width: 50; height: 25
            color: "lightblue"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 300; y: 100
            width: 50; height: 25
            color: "red"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 300; y: 300
            width: 50; height: 25
            color: "green"
        }
        Rectangle {
            parent: navigable.containerItem
            x: 100; y: 300
            width: 50; height: 25
            color: "blue"
        }
        */
   

        RowLayout {
            CheckBox {
                text: "Grid Visible"
                enabled: navigable.grid
                checked: navigable.grid ? navigable.grid.visible : false
                onCheckedChanged: navigable.grid.visible = checked
            }
            Label { text: "Grid Type:" }
            ComboBox {
                id: gridType
                textRole: "key"
                model: ListModel {
                    ListElement { key: "Lines";  value: 25 }
                    ListElement { key: "None"; value: 50 }
                }
                currentIndex: 0 // Default to "Lines"
                onActivated: {
                    switch ( currentIndex ) {
                    case 0: navigable.grid = lineGrid; break;
                    case 2: navigable.grid = null; break;
                    }
                }
            }
            Label { text: "Grid Scale:" }
            ComboBox {
                textRole: "key"
                model: ListModel {
                    ListElement { key: "25";    value: 25 }
                    ListElement { key: "50";    value: 50 }
                    ListElement { key: "100";   value: 100 }
                    ListElement { key: "150";   value: 150 }
                }
                currentIndex: 1 // Default to 100
                onActivated: {
                    var gridScale = model.get(currentIndex).value
                    if ( gridScale )
                        navigable.grid.gridScale = gridScale
                }
            }
            Label { Layout.leftMargin: 25; text: "Grid Major:" }
            SpinBox {
                from: 1;    to: 10
                enabled: navigable.grid
                value: navigable.grid ? navigable.grid.gridMajor : 0
                onValueModified: navigable.grid.gridMajor = value
            }
            Label { Layout.leftMargin: 25; text: "Point size:" }
            SpinBox {
                from: 1;    to: 10
                enabled: navigable.grid
                value: navigable.grid ? navigable.grid.gridWidth : 0
                onValueModified: navigable.grid.gridWidth = value
            }
        }
}
