/*
 Copyright (c) 2008-2023, Benoit AUTHEMAN All rights reserved.

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

import QtQuick                   2.12
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import zeno.enum 1.0

Qan.NodeItem {
    id: nodeItem
    //width: 110; height: 60
    //x: 15;      y: 15

    // 自动适应子布局的尺寸
    implicitWidth: mainmain_layout.implicitWidth
    implicitHeight: mainmain_layout.implicitHeight;
    resizable: false

    property alias isview : view_btn.checked
    property alias isbypass: bypass_btn.checked
    property alias isnocache: nocache_btn.checked
    property alias isclearsubnet: clearcache_btn.checked
    property bool isloaded : true
    property var node_type : NodeType.Node_Normal
    property bool is_subnet_or_asset: false
    property bool is_locked: false
    property var node_enable_bg_color: "#0277D1"
    property var nodestatus: 0

    property string defattr: "abc"
    property string clsname : ""

    //clip: true        //设置会导致选框不出现

    readonly property real backRadius: nodeItem && nodeItem.style ? nodeItem.style.backRadius : 4.    

    function getNearestSocket(group, pos, containeritem) {
        //pos是grid层面的pos
        var paramsgroup = (group == ParamGroup.InputObject) ? inputobjparams : outputobjparams;
        var min_dist = 10000000
        var nearest_obj = null
        for (var i = 0; i < paramsgroup.count; i++) {
            var instZObjSock = paramsgroup.itemAt(i)
            if (instZObjSock.visible) {
                var sockpos = containeritem.mapFromGlobal(instZObjSock.mapToGlobal(Qt.point(instZObjSock.width/2, instZObjSock.height/2)))
                let distance = Math.sqrt(Math.pow(sockpos.x - pos.x, 2) + Math.pow(sockpos.y - pos.y, 2));
                if (distance < min_dist) {
                    min_dist = distance
                    nearest_obj = instZObjSock
                }
            }
        }
        return nearest_obj
    }

    function getSocketObject(paramName, group) {
        if (group == ParamGroup.InputObject) {
            for (let i = 0; i < inputsocks_layout.children.length; i++) {
                let instZObjSock = inputsocks_layout.children[i];
                if (instZObjSock.socket_name == paramName) {
                    return instZObjSock
                }
            }
            return null
        }
        else if (group == ParamGroup.InputPrimitive) {
            for (let i = 0; i < nodebody_layout.children.length; i++) {
                let child = nodebody_layout.children[i]
                if (child.name == paramName && child.group == group) {
                    return child.sockid
                }
            }
            return null
        }
        else if (group == ParamGroup.OutputPrimitive) {
            for (let i = 0; i < nodebody_layout.children.length; i++) {
                let child = nodebody_layout.children[i]
                if (child.name == paramName && child.group == group) {
                    return child.sockid
                }
            }
            return null
        }
        else if (group == ParamGroup.OutputObject) {
            for (let i = 0; i < outputsocks_layout.children.length; i++) {
                let instZObjSock = outputsocks_layout.children[i];
                if (instZObjSock.socket_name == paramName) {
                    return instZObjSock
                }
            }
            return null
        }
    }

    onIsviewChanged: function() {
        var graphM = nodeItem.graph.model;
        var idx = nodeItem.node.index;
        graphM.setData(idx, nodeItem.isview, Model.ROLE_NODE_ISVIEW)
        // console.log("onIsviewChanged: " + nodeItem.isview)
    }

    onIsbypassChanged: function() {
        var graphM = nodeItem.graph.model;
        var idx = nodeItem.node.index;
        graphM.setData(idx, nodeItem.isbypass, Model.ROLE_NODE_BYPASS)
        //console.log("onIsbypassChanged: " + nodeItem.isbypass)
    }

    onIsnocacheChanged: function() {
        var graphM = nodeItem.graph.model
        var idx = nodeItem.node.index
        graphM.setData(idx, nodeItem.isnocache, Model.ROLE_NODE_NOCACHE)
    }

    onIsclearsubnetChanged: function() {
        var graphM = nodeItem.graph.model
        var idx = nodeItem.node.index
        graphM.setData(idx, nodeItem.isclearsubnet, Model.ROLE_NODE_CLEARSUBNET)
    }

    onDataChanged: function(data, role) {
        //console.log(nodeItem.node.label + " onDataChanged: " + data + ", role = " + role)
        if (role == Model.ROLE_NODE_ISVIEW) {
            //console.log("data = " + data)
            nodeItem.isview = data
        }
        else if (role == Model.ROLE_NODE_BYPASS) {
            nodeItem.isbypass = data
        }
        else if (role == Model.ROLE_NODE_CLEARSUBNET) {
            nodeItem.isclearsubnet = data
        }
        else if (role == Model.ROLE_NODE_NOCACHE) {
            nodeItem.isnocache = data
        }
        else if (role == Model.ROLE_OBJPOS) {
            nodeItem.x = data.x
            nodeItem.y = data.y           
        }
        else if (role == Model.ROLE_NODE_LOCKED) {
            nodeItem.is_locked = data
        }
        else if (role == Model.ROLE_NODE_RUN_STATE) {
            console.log("ROLE_NODE_RUN_STATE change, data = " + data)
            nodeItem.nodestatus = data
        }
        else if (role == Model.ROLE_NODE_IS_LOADED) {
            //console.log("onDataChanged: Model.ROLE_NODE_IS_LOADED")
            nodeItem.isloaded = data
            if (nodeItem.isloaded) {
                //console.log("nodeItem.isloaded")
                roundrectheader.bgcolor = rectheader.color = nodeItem.node_enable_bg_color
            }
            else {
                //console.log("nodeItem.isloaded")
                roundrectheader.bgcolor = rectheader.color = "#5F5F5F"
            }
        }
    }

    //event:
    onNodeDoubleClicked: function(node, pos) {
        var graphM = nodeItem.graph.model;
        var idx = nodeItem.node.index;
        var nodetype = graphM.data(idx, Model.ROLE_NODETYPE);       //ROLE_OBJPOS
        if (nodetype >= 5) {
            //子图
            var graph_path = graphM.path()
            var nodename = graphM.data(idx, Model.ROLE_NODE_NAME)   //ROLE_NODE_NAME
            var is_asset_ref = graphM.data(idx, Model.ROLE_NODETYPE) == NodeType.Node_AssetReference
            graph_path.push(nodename)
            if (is_asset_ref) {
                graphs_stack.jumpToAsset(nodeItem.clsname)
            }
            else {
                graphs_stack.stepIntoSubnet(graph_path)
            }
        } else {
            var pos2 = nodename_editor.mapFromItem(nodeItem, pos)
            if (pos2.x > 0 && pos2.y > 0 &&
                pos2.x < nodename_editor.width &&
                pos2.y < nodename_editor.height) {
                    if(nodeItem.clsname !== "SubInput" && nodeItem.clsname !== "SubOutput"){
                        nodename_editor.isEditing = true
                    }
            }
        }
    }

    Column {
        id: detach_name_editor
        anchors.right: nodeItem.left
        y:0//y: right_status_group.y + right_status_group.height / 2 - height / 2

        EditableText {
            //anchors.verticalCenter: right_status_group.verticalCenter
            //anchors.verticalCenter: mainmain_layout.verticalCenter
            text: nodeItem.node.label
            fontsize: 22
            handle_mouseevent : nodeItem.clsname !== "SubInput" && nodeItem.clsname !== "SubOutput"
            onTextChanged: {
                nodeItem.node.label = text
            }
        }
        Row {
            Text {
                color: "gray"
                text: nodeItem.clsname
                font.pixelSize: 16
                Layout.alignment: Qt.AlignVCenter
            }

            Image {
                id: lockimg
                visible: {
                    return nodeItem.node_type == NodeType.Node_AssetInstance || 
                        nodeItem.node_type == NodeType.Node_AssetReference
                }
                width: 16
                height: 16
                z: 10
                source: {
                    if (nodeItem.node_type != NodeType.Node_AssetInstance && nodeItem.node_type != NodeType.Node_AssetReference) {
                        return ""
                    }
                    return nodeItem.is_locked ? "qrc:/icons/lock.svg" : "qrc:/icons/unlock.svg"
                }
            }
        }
    }


    Image {
        id: errormark
        visible: nodeItem.nodestatus == NodeStatus.RunError
        width: 64
        height: 64
        z: -10
        source: "qrc:/icons/node/error.svg"
        anchors.right: nodeItem.left
        y: right_status_group.y + right_status_group.height / 2 - height / 2
    }

    ColumnLayout {
        id: mainmain_layout
        spacing: 1

        RowLayout {
            id: inputsocks_layout

            Item {
                Layout.fillWidth: true
            }

            Repeater {
                id: inputobjparams
                model: nodeItem.node.params.inputObjects()

                delegate: 
                    ZObjSocket {
                        id: input_obj_socket
                        required property string name
                        required property string nodename
                        required property int group
                        required property color socket_color

                        socket_name: name
                        socket_group: group
                        bg_color: socket_color

                        visible: true

                        Layout.preferredHeight: height
                        Layout.preferredWidth: width

                        onSocketClicked: function() {
                            var centerpos = Qt.point(input_obj_socket.width / 2, input_obj_socket.height / 2)
                            var globalPosition = input_obj_socket.mapToGlobal(centerpos)
                            nodeItem.socketClicked(nodeItem, ParamGroup.InputObject, socket_name, globalPosition)
                        }
                    }
            }

            Item {
                Layout.fillWidth: true
            }
        }

        RowLayout {
            //为了使节点框内的布局居中，所以在外面再套一个横向布局
            id: title_prims_layout

            Item {
                Layout.fillWidth: true
            }        

            Item {
                id: main_item
                implicitWidth: main_layout.implicitWidth
                implicitHeight: main_layout.implicitHeight

                // 阴影，因性能开销暂时不用
                // Loader {
                //     id: delegateLoader
                //     anchors.fill: parent
                //     source: "qrc:/QuickQanava/RectSolidShadowBackground.qml"

                //     onItemChanged: {
                //         if (item)
                //             item.style = nodeItem.style
                //     }
                // }

                ColumnLayout {
                    id: main_layout
                    anchors.margins: 0
                    spacing: 0

                    Item {
                        id: nodeheader
                        implicitWidth: nodeheader_layout.implicitWidth
                        implicitHeight: 40//nodeheader_layout.implicitHeight
                        Layout.fillWidth: true

                        /*以下两个rect各选一，视乎数值参数是否隐藏*/
                        ZRoundRect {
                            id: roundrectheader
                            anchors.fill: parent
                            bgcolor: nodeItem.node_enable_bg_color //"#0277D1"
                            radius: nodeItem.backRadius
                            visible: nodebody.visible
                        }

                        Rectangle {
                            id: rectheader
                            anchors.fill: parent
                            color: nodeItem.node_enable_bg_color  //"#0277D1"
                            radius: nodeItem.backRadius
                            visible: !nodebody.visible
                        }

                        RowLayout {
                            id: nodeheader_layout
                            Layout.fillWidth: true
                            spacing: 0

                            Item {
                                id: left_status_group
                                width: {
                                    //is_subnet_or_asset可能还没更新导致第一次判断不对，直接用内置属性
                                    var type = nodeItem.node.nodeType
                                    if (type == NodeType.Node_SubgraphNode ||
                                        type == NodeType.Node_AssetInstance ||
                                        type == NodeType.Node_AssetReference)
                                    {
                                        return nocache_btn.width + clearcache_btn.width - 4/*xoffset*/ + 1/*space*/
                                    }
                                    else {
                                        return nocache_btn.width
                                    }
                                }
                                height: 40

                                StatusRoundBtnLeft {
                                    id: nocache_btn
                                    anchors.left: parent.left
                                    anchors.top: parent.top
                                    basefillcolor: "#FF3300"
                                    radius: nodeItem.backRadius - 1
                                    is_round_bottom: !nodebody.visible
                                    height: 40
                                    z: 100
                                }

                                StatusSquareBtn {
                                    id: clearcache_btn
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                    basefillcolor: "#E302F8"
                                    is_left: true
                                    visible: nodeItem.is_subnet_or_asset
                                    height: 40
                                    z: 100
                                }
                            }

                            RowLayout {
                                id: nodenamelayout
                                Layout.fillWidth: true

                                Rectangle {
                                    id: left_block
                                    width: 18
                                    Layout.fillHeight: true
                                    color: "transparent"
                                }

                                EditableText {
                                    id: nodename_editor
                                    Layout.fillWidth: true
                                    //horizontalAlignment: Text.AlignHCenter
                                    text: nodeItem.node.label
                                    handle_mouseevent: false

                                    onTextChanged: {
                                        nodeItem.node.label = text
                                    }
                                }

                                Image {
                                    id: nodeicon
                                    width: 20
                                    height: 20
                                    //source: "data:image/svg+xml;utf8," + ""
                                }    

                                Rectangle {
                                    id: right_block
                                    width: 18
                                    Layout.fillHeight: true
                                    color: "transparent"
                                }
                            }

                            Item {
                                id: right_status_group
                                width: view_btn.width + bypass_btn.width - 4/*xoffset*/ + 1/*space*/
                                height: 40

                                StatusSquareBtn {
                                    id: bypass_btn
                                    basefillcolor: "#FFBD21"
                                    is_left: false
                                    height: 40
                                    z: 100
                                    anchors.left: parent.left
                                    anchors.top: parent.top
                                }

                                StatusRoundBtnRight {
                                    id: view_btn
                                    basefillcolor: "#30BDD4"
                                    height: 40
                                    radius: nodeItem.backRadius - 1
                                    is_round_bottom: !nodebody.visible
                                    z: 100
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                }
                            }
                        }
                    }

                    Item {
                        id: nodebody
                        implicitWidth: nodebody_layout.implicitWidth
                        implicitHeight: nodebody_layout.implicitHeight
                        Layout.fillWidth: true
                        visible: nodeItem.node.params.showPrimSocks;

                        ZRoundRect {
                            anchors.fill: parent
                            bgcolor: "#303030"
                            radius: nodeItem.backRadius
                            top_radius: false
                        }

                        ColumnLayout {
                            id: nodebody_layout
                            anchors.fill: parent

                            Item {
                                id: just_top_margin
                                height: 5
                            }

                            Repeater{
                                id: inputprimparams
                                model: nodeItem.node.params.inputPrims()

                                delegate:
                                    Text {
                                        required property string name
                                        required property int group
                                        required property bool socket_visible
                                        readonly property int hmargin: 10
                                        property alias sockid: input_prim_socket

                                        visible: socket_visible

                                        color: "white"
                                        text: name
                                        Layout.alignment: Qt.AlignLeft
                                        Layout.leftMargin: hmargin

                                        Rectangle {
                                            id: input_prim_socket
                                            height: parent.height * 0.8
                                            width: height
                                            radius: height / 2
                                            color: "#CCA44E"
                                            x: -parent.hmargin - width/2.
                                            anchors.verticalCenter: parent.verticalCenter

                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: function() {
                                                    var centerpos = Qt.point(input_prim_socket.width / 2, input_prim_socket.height / 2)
                                                    var globalPosition = input_prim_socket.mapToGlobal(centerpos)
                                                    nodeItem.socketClicked(nodeItem, ParamGroup.InputPrimitive, name, globalPosition)
                                                }
                                            }
                                        }
                                    }
                            }

                            Repeater {
                                id: outputprimparams
                                model: nodeItem.node.params.outputPrims()

                                delegate:
                                    Text {
                                        required property string name
                                        required property int group
                                        required property bool socket_visible
                                        readonly property int hmargin: 10
                                        property alias sockid: output_prim_socket

                                        visible: socket_visible

                                        color: "white"
                                        text: name
                                        Layout.alignment: Qt.AlignRight
                                        Layout.rightMargin: 10

                                        Rectangle {
                                            id: output_prim_socket
                                            height: parent.height * 0.8
                                            width: height
                                            radius: height / 2
                                            color: "#CCA44E"
                                            x: parent.width + parent.hmargin - width/2.
                                            anchors.verticalCenter: parent.verticalCenter
                                            
                                            MouseArea {
                                                anchors.fill: parent
                                                onClicked: function() {
                                                    var centerpos = Qt.point(output_prim_socket.width / 2, output_prim_socket.height / 2)
                                                    var globalPosition = output_prim_socket.mapToGlobal(centerpos)
                                                    nodeItem.socketClicked(nodeItem, ParamGroup.OutputPrimitive, name, globalPosition)
                                                }
                                            }
                                        }
                                    }
                            }                          

                            Item {
                                id: just_bottom_margin
                                height: 5
                            }
                        }
                    }
                }
            }

            Item {
                Layout.fillWidth: true
            }
        }

        Rectangle {
            id: dirtymark
            color: "yellow"
            height: 2
            width: 48
            visible: nodeItem.nodestatus != NodeStatus.RunSucceed
            Layout.alignment: Qt.AlignHCenter
        }

        RowLayout {
            id: outputsocks_layout

            Item {
                Layout.fillWidth: true
            }

            Repeater{
                id: outputobjparams
                model: nodeItem.node.params.outputObjects()

                delegate: ZObjSocket {
                    id: output_obj_socket
                    required property string nodename
                    required property string name
                    required property int group
                    required property color socket_color

                    Layout.preferredHeight: height
                    Layout.preferredWidth: width

                    socket_name: name
                    socket_group: group
                    bg_color: socket_color
                    visible: true
                    onSocketClicked: function() {
                        var centerpos = Qt.point(output_obj_socket.width / 2, output_obj_socket.height / 2)
                        var globalPosition = output_obj_socket.mapToGlobal(centerpos)
                        nodeItem.socketClicked(nodeItem, ParamGroup.OutputObject, socket_name, globalPosition)
                    }
                }
            }

            Item {
                Layout.fillWidth: true
            }
        }
    }

    Component.onCompleted: {
        //初始化基本数据:
        var graphM = nodeItem.graph.model;
        var idx = nodeItem.node.index;
        var nodename = graphM.data(idx, Model.ROLE_NODE_NAME)   //ROLE_NODE_NAME
        var pos = graphM.data(idx, Model.ROLE_OBJPOS);       //ROLE_OBJPOS
        nodeItem.x = pos.x
        nodeItem.y = pos.y
        nodeItem.isview = graphM.data(idx, Model.ROLE_NODE_ISVIEW)
        nodeItem.isnocache = graphM.data(idx, Model.ROLE_NODE_NOCACHE)
        nodeItem.isclearsubnet = graphM.data(idx, Model.ROLE_NODE_CLEARSUBNET)
        nodeItem.clsname = graphM.data(idx, Model.ROLE_CLASS_NAME)
        nodeItem.isloaded = graphM.data(idx, Model.ROLE_NODE_IS_LOADED)
        nodeItem.node_type = graphM.data(idx, Model.ROLE_NODETYPE)
        nodeItem.is_locked = graphM.data(idx, Model.ROLE_NODE_LOCKED)
        nodeItem.nodestatus = graphM.data(idx, Model.ROLE_NODE_RUN_STATE)
        nodeItem.is_subnet_or_asset = nodeItem.node_type == NodeType.Node_SubgraphNode || nodeItem.node_type == NodeType.Node_AssetInstance || nodeItem.node_type == NodeType.Node_AssetReference

        var uistyle = graphM.data(idx, Model.ROLE_NODE_UISTYLE)
        if (uistyle["icon"] != "") {
            nodeicon.visible = true
            detach_name_editor.visible = true
            nodename_editor.visible = false
            nodeicon.source = "data:image/svg+xml;utf8," + uistyle["icon"]
        }
        else {
            nodeicon.visible = false
            detach_name_editor.visible = false
            nodename_editor.visible = true
        }
        //console.log("ui.icon = " + uistyle["icon"])
        if (uistyle["background"] != "") {
            nodeItem.node_enable_bg_color = uistyle["background"]
            //roundrectheader.bgcolor = rectheader.color = uistyle["background"];
        }
        if (!nodeItem.isloaded) {
            nodeItem.node_enable_bg_color = "#5F5F5F"
        }
    }
}