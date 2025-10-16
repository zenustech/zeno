import QtQuick                   2.8
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import Qt.labs.platform          1.0    // ColorDialog

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import "." as Qan

Qan.NodeItem {
    id: it
    width: 340
    //height: 300
    resizable: false
    connectable : Qan.NodeItem.UnConnectable
    property int headerHeight: 75
    property int portHeight

    property var borderColor: "#121416"
    property var borderWidth: 2
    property var headerColor: "#246283"
    property var bodyColor: "#303030"
    property var componentBgClr: "#191D21"
    property var textClr: "#FFFFFF"
    
    leftDock: Qan.VerticalDock {
        hostNodeItem: it
        //height:body.height   //配合y:body.y可以使其高的上下限和body匹配
        //y:body.y
    }
    rightDock: Qan.VerticalDock {
        hostNodeItem: it
    }
    Rectangle{
        anchors.fill:parent
        border.color: it.borderColor; border.width:it.borderWidth
        Column{
            id:colLayout
            width: parent.width - it.borderWidth*2
            height: parent.height - it.borderWidth*2
            anchors.centerIn: parent
            anchors.top:parent.top
            Rectangle{
                id:header
                width:parent.width
                height:it.headerHeight
                //anchors.top: parent.top
                //width:parent.width
                //height:parent.height/4
                color: it.headerColor
                //RowLayout{
                Row{
                    id:headerLayout
                    anchors.fill: parent
                    layoutDirection: "RightToLeft"
                    rightPadding: 0
                    leftPadding: 20
                    spacing:0
                    Rectangle {
                        id:btn1
                        width:30
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        //anchors.rightMargin: 10
                        //Layout.rightMargin:10
                        color: "#30BDD4"
                        //Label {id:li; width:50; anchors.centerIn: parent}
                    }
                    Rectangle {
                        id:split1
                        width:2
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        color: "#484848"
                    }
                    Rectangle {
                        id:btn2
                        width:30
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        //anchors.rightMargin: 10
                        //Layout.rightMargin:10
                        color: "#2E313A" //"#E302F8"
                        //Label { anchors.centerIn: parent}
                    }
                    Rectangle {
                        id:split2
                        width:2
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        color: "#484848"
                    }
                    Rectangle {
                        id:btn3
                        width:30
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        //anchors.rightMargin: 10
                        //Layout.rightMargin:10
                        color: "#2E313A" //"#FFBD21"
                        //Label {anchors.centerIn: parent}
                    }
                    Rectangle {
                        id:lblRec
                        width:parent.width - btn1.width*3 - split1.width*2
                        height:parent.height
                        anchors.verticalCenter: parent.verticalCenter
                        //anchors.rightMargin: 10
                        //Layout.rightMargin:10

                        //z:-1
                        color: "transparent"
                        Label {id: nodeName; text: "CustomNode";anchors.fill:parent;horizontalAlignment: Text.AlignHCenter;verticalAlignment: Text.AlignVCenter; font.pixelSize: 18; color:"#FFFFFF"}
                    }
                }
            }
            Rectangle{
                id:split
                width:parent.width
                height:it.borderWidth
                color: it.borderColor
            }
            Rectangle{
                id:body
                width:parent.width
                height:parent.height - it.headerHeight - split.height
                //anchors.top: header.bottom
                //width:parent.width
                //height:parent.height - header.height
                color: it.bodyColor
                Column {
                    id:bodyLayout
                    anchors.fill: parent
                    spacing: 6
                    topPadding: 10
                    bottomPadding: 10

                    //color: "#2D3239"
                    //color:"lightblue"
                    //radius: 0
                    //border.color: "green"; border.width: 3
                    //Label { anchors.centerIn: parent; text: "CUSTOM" }
                }
            }
        }
    }


    function createWidget(name){
        var component
        if(name == "text"){
            component = Qt.createComponent("MySingleText.qml")
        }else if(name == "textedit"){
            component = Qt.createComponent("MyMultiText.qml")
        }else if(name == "slider"){
            component = Qt.createComponent("MySlider.qml")
        }else if(name == "checkbox"){
            component = Qt.createComponent("MyCheckBox.qml")
        }else if(name == "combobox"){
            component = Qt.createComponent("MyCheckBox.qml")
        }else if(name == "fileinput"){
            component = Qt.createComponent("MyFileDialog.qml")
        }else if(name == "vec2text"){
            component = Qt.createComponent("MyVec2Text.qml")
        }else if(name == "vec3text"){
            component = Qt.createComponent("MyVec3Text.qml")
        }else if(name == "vec4text"){
            component = Qt.createComponent("MyVec4Text.qml")
        }
        return component.createObject(bodyLayout)
    }

    function addComponent(topo, node, componentLst, portNameLst, portTypeLst, nodeName){
        const componentWidth = bodyLayout.width*0.5
        const componentHeight = 30
        const componentTopMargin = 6
        var bodyHeight = 0
        var componentStartY = header.y + header.height + bodyLayout.topPadding

        for(let i = 0; i < componentLst.length; i++){
            var obj = createWidget(componentLst[i])
            obj.width = componentWidth
            //obj.anchors.topMargin = 100
            obj.anchors.horizontalCenter = bodyLayout.horizontalCenter

            if(componentLst[i] == "textedit"){
                bodyHeight += 100 + componentTopMargin
                obj.height = 100
            }else{
                bodyHeight += componentHeight + componentTopMargin
                obj.height = componentHeight
            }
            obj.backGroundColor = it.componentBgClr
            /*if(i == 0){
                obj.anchors.top = bodyLayout.top
            }else{
                obj.anchors.top = topComp.bottom
            }*/

            if(portTypeLst[i] == "in"){
                var port = topo.insertPort(node, Qan.NodeItem.Left, Qan.PortItem.In, portNameLst[i], nodeName + portNameLst[i])
            }else{
                var port = topo.insertPort(node, Qan.NodeItem.Right, Qan.PortItem.Out, portNameLst[i], nodeName + portNameLst[i])
            }
            port.label = portNameLst[i]
            //port.txt = portNameLst[i]
            port.multiplicity = Qan.PortItem.Single

            var curComponentHeight = componentLst[i] == "textedit" ? 100 : componentHeight
            port.y = componentStartY + (curComponentHeight - portHeight) / 2 + it.borderWidth * 2
            componentStartY += curComponentHeight + componentTopMargin
        }
        it.height = bodyHeight + bodyLayout.bottomPadding + it.headerHeight + split.height*3
    }

    function initializeNode(topology, n1, nodeData){
        n1.label = nodeData.name
        n1.item.x = nodeData.x
        n1.item.y = nodeData.y
        //nodeName.text = nodeData.name
        addComponent(topology, n1, nodeData.widgetLst, nodeData.portNameLst, nodeData.portTypeLst, nodeData.name)
    }
}
