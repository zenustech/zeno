import QtQuick 2.12
import QtQuick.Layouts 1.3
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import zeno.enum 1.0


Rectangle {
    id: qmlnode
    property string arg_ident
    property string arg_name     //arg_name是为了避免和model导出的name重名
    property variant paramModel
    property variant subgModel
    property int repeaterIndex
    
    property var sockOnClicked
    property var mismatchSocket
    property var matchSocket
    property var destoryTempEdge
    property var addLink
    property var getTempEdge

    color: "#303030"

    signal showNodeMenu(var qmlnode, var pos)

    CustomBorder
    {
        commonBorderWidth: 2
        borderColor: "black"
    }

    implicitWidth:  mainLayout.implicitWidth
    implicitHeight: mainLayout.implicitHeight

    MouseArea {
        id: mouseArea1
        anchors.fill: parent
        drag.target: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onPressed: {
            //qmlnode.beginDrag = Qt.point(qmlnode.x, qmlnode.y);
        }
        onReleased: {
            
        }
        onClicked: {
            //console.log("<--------click node----------->")
            if (mouse.button == Qt.RightButton) {
                showNodeMenu(qmlnode, Qt.point(mouse.x, mouse.y))
            }
            qmlnode.forceActiveFocus()  //make all textInput focus out
            
            var edge = qmlnode.getTempEdge()
            if (edge.visible && edge.nodeId != arg_ident && edge.isMatch) {
                var socketobj = params.getNearSocket(Qt.point(mouse.x, mouse.y), !edge.isFromInput);
                if (socketobj != null)
                    qmlnode.addLink(socketobj)
            }
            else {
                qmlnode.destoryTempEdge();
            }
        }

        onMouseYChanged: {
            var edge = qmlnode.getTempEdge()
            //console.log("edge.nodeId = " + edge.nodeId + "arg_ident = " + arg_ident)
            if (edge.visible && edge.nodeId != arg_ident) {   
                var socketobj = params.getNearSocket(Qt.point(mouse.x, mouse.y), !edge.isFromInput);
                if (socketobj != null)
                   qmlnode.matchSocket(socketobj)
            }
            else if(edge.visible){
                qmlnode.mismatchSocket()
            }
        }

        onExited: {
             if (qmlnode.getTempEdge().visible)
                qmlnode.mismatchSocket()
        }

        ColumnLayout  {
            id: mainLayout
            spacing: 0
            anchors.fill: parent

            Rectangle {
                id: node_header
                color: qmlnode.subgModel != null ? "#1D5F51" : "#246283"
                implicitWidth: header_layout.implicitWidth
                implicitHeight: 66
                Layout.fillWidth: true

                RowLayout {
                    id: header_layout
                    anchors.fill: parent

                    FixSpacer { width: 24; height: 1 }

                    Text {
                       id: btnshowparams
                       text: qmlnode.arg_name
                       font.family: "Consolas"
                       font.pixelSize: 20;
                       font.bold: true;
                       color: "white"
                    }

                    FillSpacer {}

                    StatusBtnGroup {

                    }
                }
            }

            FixSpacer { width: 1; height: 16 }

            ColumnLayout {
                id: bodyLayout
                anchors.margins: 8

                Repeater{
                    id: params
                    model: qmlnode.paramModel

                    function getZParam(name, isinput) {
                        var idx = qmlnode.paramModel.indexFromName(name, isinput)
                        if (idx != -1) {
                            //console.log(idx)
                            return params.itemAt(idx)
                        } else {
                            return null
                        }
                    }

                    function getNearSocket(point, isinput) {
                        var idxList = qmlnode.paramModel.getIndexList(isinput)
                        if (idxList != undefined && idxList.length > 0){
                            var nearSockObj = params.itemAt(idxList[0]).getSocketItemObj();
                            var nearPos = mouseArea1.mapFromItem(nearSockObj, 0, 0)
                            for (var i = 1; i < idxList.length; ++i ){
                                var socketObj = params.itemAt(idxList[i]).getSocketItemObj()
                                var sockNodePos = mouseArea1.mapFromItem(socketObj, 0, 0)
                                if (Math.abs(nearPos.y - point.y)  > Math.abs(sockNodePos.y - point.y)){
                                    nearSockObj = socketObj
                                    nearPos = sockNodePos
                                }
                            }
                            return nearSockObj
                        }
                        else{
                            return null
                        }
                    }

                    delegate: ZParam {
                        required property string name
                        required property string type
                        required property bool input
                        required property int control

                        arg_name:    name
                        arg_isinput: input
                        arg_control: control
                        sockOnClicked: qmlnode.sockOnClicked
                        mismatchSocket: qmlnode.mismatchSocket
                        matchSocket: qmlnode.matchSocket
                    }
                }
            }

            //bottom space
            FixSpacer { width: 1; height: 16 }
        }
    }

    function getSocketPos(paramName, isinput) {
        var zparamobj = params.getZParam(paramName, isinput)
        var socketpos = zparamobj.getSocketPos()
        //console.log('socketpos', socketpos.x, socketpos.y)
        return {'x': qmlnode.x + socketpos.x, 'y': qmlnode.y + socketpos.y}
    }

    function getSocketGlobalPos(paramName, isinput) {
        var zparamobj = params.getZParam(paramName, isinput)
        var socketobj = zparamobj.getSocketItemObj()
        return socketobj.mapToItem(null, 0)
    }

    function getSocketObj(paramName, isinput) {
        return params.getZParam(paramName, isinput).getSocketItemObj()
    }

    Component.onCompleted: {
    }

    Connections {
        /*
        target: thisnode
        function onNewparamCommand()
        {
            //params.add(new ZParam {name:"param3"})
            console.log("onNewparamCommand")
        }
        */
    }
}
