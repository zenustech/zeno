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
import QtQuick.Controls          2.3
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import zeno.enum 1.0
import Zeno 1.0

Item {
    id: root
    property variant graphModel

    signal navigateRequest(var lst)
    signal nodeClicked(var node)

    ToolTip { id: toolTip; timeout: 2500 }
    

    Qan.GraphView {
        id: graphView
        navigable   : true
        resizeHandlerColor: "#03a9f4"       // SAMPLE: Set resize handler color to blue for 'resizable' nodes
        gridThickColor: Qt.rgba(46/255, 46/255, 46/255, 1.0)// Material.theme === Material.Dark ? "#4e4e4e" : "#c1c1c1"
        grid: null      //不需要grid
        focus: true
        anchors.fill: parent

        // Activate multisampling for edges antialiasing
        layer.enabled: true     //居然只要在这里设置，就可以使得边，按钮的锯齿问题一并解决了
        layer.samples: 8



        PinchHandler {
            target: null
            onActiveScaleChanged: {
                console.error('centroid.position=' + centroid.position)
                console.error('activeScale=' + activeScale)
                var p = centroid.position
                var f = activeScale > 1.0 ? 1. : -1.
                graphView.zoomOn(p, graphView.zoom + (f * 0.03))
            }
        }

        property var tempEdge: undefined    //由于tempEdge需要挂在containerItem下，所以不能以组件的方式直接定义

            // 背景透明层
        Rectangle {
            id: backgroundOverlay
            anchors.fill: parent
            color: "#25FFFFFF"
            z: 999
            visible: graph.locked
        }

        Dialog {
            id: listkeydialog
            modal: true
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1000
            standardButtons: Dialog.NoButton
            width: 300
            height: 150
            focus: true

            property var _resolve

            contentItem: Column {
                spacing: 20
                padding: 20

                Text {
                    text: "是否作为子元素加入 List？"
                    font.pointSize: 12
                    wrapMode: Text.Wrap
                }

                Row {
                    spacing: 10

                    Button {
                        text: "作为子元素"
                        onClicked: {
                            listkeydialog._resolve(true)
                            listkeydialog.close()
                        }
                    }

                    Button {
                        text: "作为整体"
                        onClicked: {
                            listkeydialog._resolve(false)
                            listkeydialog.close()
                        }
                    }
                }
            }
        }

        // 包装一个返回 Promise 的函数
        function askUserChoice() {
            return new Promise(function(resolve, reject) {
                listkeydialog.open()
                listkeydialog._resolve = resolve
            })
        }

        //background
        Rectangle {
            anchors.fill: parent
            color: {
                //if (graph.locked)
                //    return Qt.rgba(83/255, 83/255, 85/255, 0.3)
                //else
                    return Qt.rgba(26/255, 26/255, 26/255, 1)
            }
            z: -10
        }

        Component {
            id: tmp_edge_component
            Edge {
                visible: true
                color: "#5FD2FF"
                thickness: 2

                /*
                //此方式没法动态更新，因为没法捕获globalpos的changed信号，必须要加上hoverInfo才能捕获，干脆在onHoverInfoChanged做好了
                //除非以后想到如何动态捕获鼠标的变化（不需要添加额外的MouseArea为前提）
                Component.onCompleted: {
                    point2x = Qt.binding(function() {
                        graphView.hoverInfo;
                        var mousepos = graphView.containerItem.mapFromGlobal(MouseUtils.getGlobalMousePosition())
                        return mousepos.x
                    })
                    point2y = Qt.binding(function() {
                        graphView.hoverInfo;
                        var mousepos = graphView.containerItem.mapFromGlobal(MouseUtils.getGlobalMousePosition())
                        return mousepos.y
                    })
                }
                */
            }
        }

        function clear_temp_edge() {
            if (graphView.tempEdge != undefined && graphView.tempEdge.visible) {
                graphView.tempEdge.destroy()
                graphView.tempEdge = null
                graphView.tryconnect = false
            }        
        }

        function invalide_edgearea_clicked() {
            // console.log("invalide_edgearea_clicked")
            clear_temp_edge()
        }

        function temp_edge_close() {
            //!!!from指的是连线时操作的起点，不是指参数的from/output
            var from_node = tempEdge.tempedge_from_sock["node_name"]
            var from_sock = tempEdge.tempedge_from_sock["sock_name"]
            var from_group = tempEdge.tempedge_from_sock["group"]
            //console.log("from_node = " + from_node + ", from_sock = " + from_sock + ", from_group = " + from_group)

            var to_node = tempEdge.tempedge_to_sock["node_name"]
            var to_sock = tempEdge.tempedge_to_sock["sock_name"]
            var to_group = tempEdge.tempedge_to_sock["group"]
            //console.log("to_node = " + to_node + ", to_sock = " + to_sock + ", to_group = " + to_group)

            var out_node = "", in_node = "", out_sock = "", in_sock = ""
            if (from_group == ParamGroup.OutputObject || from_group == ParamGroup.OutputPrimitive) {
                out_node = from_node
                out_sock = from_sock
                in_node = to_node
                in_sock = to_sock
            }
            else {
                out_node = to_node
                out_sock = to_sock
                in_node = from_node
                in_sock = from_sock
            }

            //如果发现输入口是一个List，要询问是作为List的元素连进来，还是作为List的整体传输过来
            var out_param_type = root.graphModel.getParamType(out_node, false, out_sock)
            var in_param_type = root.graphModel.getParamType(in_node, true, in_sock)
            //console.log("in_param_type is " + in_param_type)
            //console.log("ParamType.List is " + ParamType.List)
            if (in_param_type == ParamType.List) {
                //如果输出是list或者object，要询问作为子元素还是整体
                if (out_param_type == ParamType.List || out_param_type == ParamType.IObject) {
                    askUserChoice().then(function(asListChild) {
                        graphView.graph.model.addLink(out_node, out_sock, in_node, in_sock, asListChild)
                    })
                    clear_temp_edge()
                    return
                }
            }
            graphView.graph.model.addLink(out_node, out_sock, in_node, in_sock)
            clear_temp_edge()
        }

        onClicked: {
            //console.log("Graphview.onClicked")
            clear_temp_edge()
        }

        onRightClicked: function(pos) {
            //console.log("rightclick: " + pos.x + "," + pos.y)
            newnode_menu.x = pos.x
            newnode_menu.y = pos.y
            newnode_menu.open()
        }

        onHoverInfoChanged: function() {
            var global_pos = MouseUtils.getGlobalMousePosition()
            var mousepos = graphView.containerItem.mapFromGlobal(global_pos)
            var mouse_in_view = graphView.mapFromGlobal(global_pos)
            var nodeitem = graphView.graph.nodeItemAt(mouse_in_view.x, mouse_in_view.y) // graphView.hoverInfo["node"]
            var hoverpos = mousepos

            if (nodeitem) {
                //console.log("hover node: " + nodeitem.node.label)
            }
            //console.log("hoverpos: " + hoverpos)

            if (graphView.tempEdge) {
                if (nodeitem == null || 
                        p1_group == ParamGroup.InputPrimitive ||
                        p1_group == ParamGroup.OutputPrimitive) {
                    //不处理数值类型的吸附，因为本来就不提倡这种用法
                    //console.log("no nodeitem hover")
                    graphView.tempEdge.point2x = hoverpos.x
                    graphView.tempEdge.point2y = hoverpos.y
                    graphView.tempEdge.tempedge_to_sock = null    
                    return
                }

                var nearest_sock = null
                var p1_group = graphView.tempEdge.p1_group
                var params_repeater = null
                if (p1_group == ParamGroup.InputObject) {
                    nearest_sock = nodeitem.getNearestSocket(ParamGroup.OutputObject, hoverpos, graphView.containerItem)
                }
                else if (p1_group == ParamGroup.OutputObject) {
                    nearest_sock = nodeitem.getNearestSocket(ParamGroup.InputObject, hoverpos, graphView.containerItem)
                }

                if (nearest_sock) {
                    var sock_pos_in_grid = graphView.containerItem.mapFromGlobal(nearest_sock.mapToGlobal(Qt.point(nearest_sock.width/2, nearest_sock.height/2)))
                    graphView.tempEdge.point2x = sock_pos_in_grid.x
                    graphView.tempEdge.point2y = sock_pos_in_grid.y

                    var end_sock_info = {
                        "node_name": nearest_sock.nodename,
                        "sock_name": nearest_sock.name,
                        "group": (p1_group == ParamGroup.InputObject) ? ParamGroup.OutputObject : ParamGroup.InputObject                
                    }
                    graphView.tempEdge.tempedge_to_sock = end_sock_info
                }
                else{
                    graphView.tempEdge.point2x = hoverpos.x
                    graphView.tempEdge.point2y = hoverpos.y
                    graphView.tempEdge.tempedge_to_sock = null
                }
            }
        }

        onNodeSocketClicked: function(node, group, name, socket_pos_in_grid) {
            //console.log("Qan.GraphView: node: " + node.label + ", group: " + group + ", socket_name:" + name + ",socket_pos_grid:" + socket_pos_in_grid)

            var pos_ = socket_pos_in_grid
            var is_start_to_link = false;
            if (!graphView.tempEdge) {
                graphView.tempEdge = tmp_edge_component.createObject(graphView.containerItem, {})
                is_start_to_link = true
            }

            if (is_start_to_link) {
                //初始化临时边
                graphView.tempEdge.point1x = pos_.x
                graphView.tempEdge.point1y = pos_.y
                graphView.tempEdge.p1_group = group
                graphView.tempEdge.visible = true

                var start_sock_info = {
                    "node_name": node.label,
                    "sock_name": name,
                    "group": group
                }
                graphView.tempEdge.tempedge_from_sock = start_sock_info
                graphView.tempEdge.tempedge_to_sock = null
                graphView.tryconnect = true
            } else {
                //闭合
                var end_sock_info = {
                    "node_name": node.label,
                    "sock_name": name,
                    "group": group                
                }
                graphView.tempEdge.tempedge_to_sock = end_sock_info

                graphView.temp_edge_close()
            }
        }

        onNodeClickReleased: function(node){
            proppanel.node = node
            var contentW = proppanel.calc_content_width()
            var contentH = proppanel.calc_content_height()
            proppanel.width = Math.min(graphView.width * 0.4, contentW)
            proppanel.height = Math.min(graphView.height * 0.6, contentH)
        }

        Menu {
            id: nodeMenu
            property var node: null

            MenuItem {
                text: "清空节点对象缓存"
                onTriggered: {
                    calcmgr.clearNodeObjs(nodeMenu.node.index)
                }
            }

            MenuItem {
                text: "清空子图中间结果"
                visible: {
                    if (nodeMenu.node) {
                        var model = root.graphModel
                        var nodeType = model.data(nodeMenu.node.index, Model.ROLE_NODETYPE)
                        if (nodeType == NodeType.Node_SubgraphNode || nodeType == NodeType.Node_AssetInstance) {
                            return true;
                        }
                        return false;
                    }
                    else {
                        return false
                    }
                }
                onTriggered: {
                    calcmgr.clearSubnetObjs(nodeMenu.node.index)
                }
            }

            MenuItem {
                text: "编辑自定义参数"
                height: visible ? implicitHeight : 0
                visible: {
                    if (nodeMenu.node) {
                        var model = root.graphModel
                        var nodeType = model.data(nodeMenu.node.index, Model.ROLE_NODETYPE)
                        var islocked = model.data(nodeMenu.node.index, Model.ROLE_NODE_LOCKED)
                        if (nodeType == NodeType.Node_AssetInstance) {
                            //解锁的资产才可以编辑，此时编辑的对象是这个节点自己，而不是整个资产
                            return !islocked
                        }
                        else if (nodeType == NodeType.Node_SubgraphNode) {
                            return true;
                        }
                        return false;
                    }
                    else {
                        return false
                    }
                }
                onTriggered: {
                    var nodeobj = nodeMenu.node
                    nodeMenu.close()
                    Qt.callLater(() => {   // 再延迟打开对话框
                        showDialog(nodeobj)
                    });
                    // showDialog(nodeMenu.node)
                }
            }

            MenuItem {
                text: "资产还原并锁定"
                height: visible ? implicitHeight : 0
                visible: {
                    if (nodeMenu.node) {
                        var model = root.graphModel
                        var nodeType = model.data(nodeMenu.node.index, Model.ROLE_NODETYPE)
                        var islocked = model.data(nodeMenu.node.index, Model.ROLE_NODE_LOCKED)
                        return nodeType == NodeType.Node_AssetInstance && islocked == false
                    }
                    else {
                        return false
                    }
                }
                onTriggered: {
                    var model = root.graphModel
                    model.resetAssetAndLock(nodeMenu.node.index)
                    // model.setData(nodeMenu.node.index, true, Model.ROLE_NODE_LOCKED)
                }
            }

            MenuItem {
                text: "解锁资产"
                height: visible ? implicitHeight : 0
                visible: {
                    if (nodeMenu.node) {
                        var model = root.graphModel
                        var nodeType = model.data(nodeMenu.node.index, Model.ROLE_NODETYPE)
                        var islocked = model.data(nodeMenu.node.index, Model.ROLE_NODE_LOCKED)
                        return nodeType == NodeType.Node_AssetInstance && islocked == true
                    }
                    else {
                        return false
                    }
                }
                onTriggered: {
                    var model = root.graphModel
                    model.setData(nodeMenu.node.index, false, Model.ROLE_NODE_LOCKED)
                }
            }

            MenuItem {
                text: "同步资产修改"
                height: visible ? implicitHeight : 0
                visible: {
                    if (nodeMenu.node) {
                        var model = root.graphModel
                        var nodeType = model.data(nodeMenu.node.index, Model.ROLE_NODETYPE)
                        var islocked = model.data(nodeMenu.node.index, Model.ROLE_NODE_LOCKED)
                        return nodeType == NodeType.Node_AssetInstance && islocked == false
                    }
                    else {
                        return false
                    }
                }
                onTriggered: {
                    var model = root.graphModel
                    model.syncAssetInst(nodeMenu.node.index)
                }
            }
        }

        graph: Qan.Graph {
            parent: graphView
            id: graph
            model: root.graphModel
            nodeDelegate: Qt.createComponent("qrc:/NormalNode.qml")
            selectionColor: Qt.rgba(250/255, 100/255, 0, 1.0)

            Item {
                id: nodeClickStatus
                property var clickedCount: 0
                property var node: null
                property var originX : 0
                property var originY : 0

                Timer {
                    id: singleClickTimer
                    interval: doubleClickInterval
                    running: false
                    repeat: false
                    onTriggered: {
                        var path_list = root.graphModel.path()
                        graphsmanager.onNodeSelected(path_list, nodeClickStatus.node.index)
                    }
                }
            }

            Component.onCompleted: {
                //graph.model.name()

                //var n1 = graph.insertNode()
                //n1.label = "Hello World"; n1.item.x=15; n1.item.y= 25
                //n1.item.ratio = 0.4
                //var n2 = graph.insertNode()
                //n2.label = "Node 2"; n2.item.x=15; n2.item.y= 125

                //var e = graph.insertEdge(n1, n2);
                //defaultEdgeStyle.lineType = Qan.EdgeStyle.Curved
            }
            onNodeClicked: function(node, pos) {
                if (graphView.tempEdge && graphView.tempEdge.tempedge_to_sock) {
                    graphView.temp_edge_close()
                }
                root.nodeClicked(node)

                nodeClickStatus.originX = node.item.x
                nodeClickStatus.originY = node.item.y
            }
            onNodeRightClicked: function(node) {
                var nodeType = root.graphModel.data(node.index, Model.ROLE_NODETYPE)
                var mousepos = graphView.mapFromGlobal(MouseUtils.getGlobalMousePosition())
                nodeMenu.node = node
                nodeMenu.popup(mousepos)
            }
            onNodeDoubleClicked: function(node) {
                singleClickTimer.stop()
                var nodeType = model.data(node.index, Model.ROLE_NODETYPE)
                if(nodeType != NodeType.Node_AssetInstance && nodeType != NodeType.Node_SubgraphNode) {
                    nodeClickStatus.clickedCount = 2
                }

                toolTip.text="Node <b>" + node.label + "</b> double clicked";
                toolTip.open()
                //notifyUser( "Node <b>" + node.label + "</b> double clicked" )
            }
            onNodeMoved: function(node) {
                var idx = node.index
                var x = node.item.x
                var y = node.item.y
                model.setData(idx, Qt.point(x, y), Model.ROLE_OBJPOS)
            }
            onNodeClickReleased : function(node, pos){
                if(nodeClickStatus.clickedCount != 2) {//如果是doubleclick触发第二次release，不执行
                    if(singleClickTimer.running) {
                        singleClickTimer.stop()
                    }
                    //单击时可能误移动一小段，4以内视为单击
                    if(Math.abs(node.item.x - nodeClickStatus.originX) <= 4 && Math.abs(node.item.y - nodeClickStatus.originY) <= 4) {
                        nodeClickStatus.node = node
                        singleClickTimer.start()
                    }
                }
                nodeClickStatus.clickedCount = 0
            }

            onNodeSocketClicked: function(node, group, name, socket_pos) {
            }
            onSelectedNodesChanged: {
            }
        } // Qan.Graph

        MenuEventFilter {
            id: menuKeyFilter

            onTextAppended: function(newtext) {
                searchItem.focus = true
                searchItem.forceActiveFocus()

                searchItem.text = searchItem.text + newtext
                // console.log("onTextAppended: " + searchItem.text)
            }
            onTextRemoved: function() {
                if (searchItem.text != "") {
                    searchItem.text = searchItem.text.substring(0, searchItem.text.length - 1)
                }
                searchItem.focus = true
                searchItem.forceActiveFocus()
            }
        }

        Timer {
            id: newmenuTimer
            interval: 500 // 1秒，单位为毫秒
            repeat: false
            running: false

            onTriggered: {
                //console.log("times out!")
                newnode_menu.canhighlight = true
            }
        }

        Dialog {
            id: failedDialog
            title: "create failed"
            modal: true
            standardButtons: Dialog.Ok
            width: 300
            height: 150
            onAccepted: {}

            contentItem: Text {
                text: "the module of this node has been uninstalled."
                wrapMode: Text.WordWrap
            }
        }

        Menu {
            id: newnode_menu

            property var catemenuitems: []
            property bool canhighlight: true

            implicitHeight: 300

            focus: true  // 确保 Menu 可以接收焦点

            Column {
                Rectangle {
                    width: newnode_menu.implicitWidth
                    height: 32
                    color: "#f0f0f0"

                    TextInput {
                        id: searchItem
                        anchors.fill: parent
                        anchors.margins: 8
                        focus: true // 初始设置为焦点
                        property bool clearing_text: false  //搜索完毕以后清空文本框内容

                        MouseArea {
                            anchors.fill: parent
                            onClicked: {
                                searchItem.forceActiveFocus(); // 鼠标点击时强制聚焦
                            }
                        }

                        onTextChanged: {
                            if (!clearing_text) {
                                nodecatesmodel.search(text)
                                newnode_menu.canhighlight = false
                                newmenuTimer.start()
                            }
                        }

                        Keys.onReturnPressed: {
                            if (newnode_menu.count > 1) {
                                newnode_menu.itemAt(1).triggered()
                            }
                        }

                        Keys.onPressed: {
                            if (event.key === Qt.Key_Up) {
                                
                            } else if (event.key === Qt.Key_Down) {
                                var firstItem = newnode_menu.itemAt(1)
                                firstItem.focus = true
                                firstItem.forceActiveFocus()
                            }
                        }
                    }
                }

                Text {
                    text: "Recent Search:"
                    color: "gray"
                    font.pointSize: 12
                    visible: searchItem.text == "" && !searchItem.clearing_text
                }
            }

            Component {
                id: comp_searchitem
                MenuItem {
                    id: searchresult_item
                    property bool ismenu: false
                    property bool islastitem: lastitem
                    text: name
                    property var _matchindice: keywords

                    // 动态生成带颜色的富文本字符串
                    function getColoredText(text, indices) {
                        var result = "";
                        for (var i = 0; i < text.length; i++) {
                            if (indices.indexOf(i) !== -1) {
                                result += "<span style='color:red;'>" + text[i] + "</span>";
                            } else {
                                result += text[i];
                            }
                        }
                        return result;
                    }

                    contentItem: Text {
                        textFormat: Text.RichText
                        text: getColoredText(searchresult_item.text, searchresult_item._matchindice)
                        font.pointSize: 12
                    }

                    onTriggered: {
                        //console.log("searchresult_item onTriggered")
                        var mousepos = graphView.containerItem.mapFromGlobal(MouseUtils.getGlobalMousePosition())
                        // console.log("mousepos: " + mousepos)
                        var bsucceed = nodecatesmodel.execute(graphModel, name, mousepos)
                        if (!bsucceed) {
                            console.log("!bsucceed")
                            failedDialog.open()
                        }
                    }
                }
            }

            function highlight_firstresult() {
                //确保currentIndex==1是一个有效值
                if (this.count > 1) {
                    this.currentIndex = 1
                    var item = this.itemAt(1);
                    if (item) {
                        item.highlighted = true; // 手动设置 Hover 效果
                    }
                }
            }

            Instantiator {
                model: nodecatesmodel       //global model

                delegate: comp_searchitem

                // The trick is on those two lines
                onObjectAdded: {
                    //即便我在模型端使用了beginResetModel()，QML的Instantiator还是会一个个添加
                    newnode_menu.addItem(object)
                    if (object.islastitem) {
                        newnode_menu.highlight_firstresult()
                    }
                }
                onObjectRemoved: {
                    newnode_menu.removeItem(object)
                }
            }

            onAboutToShow: {
                nodecatesmodel.init()   //会初始化整个模型，包括所有cate项，方便起见（避免后续子图加载和插件加载反复通知）
                searchItem.forceActiveFocus();
            }

            onAboutToHide: {
                searchItem.clearing_text = true
                searchItem.text = "";
                searchItem.clearing_text = false
            }

            onCurrentIndexChanged: {
                //console.log("onCurrentIndexChanged: " + newnode_menu.currentIndex)
                if (newnode_menu.currentIndex != 1) {
                    //如果第一个搜索项highlighted了，把它的highlighted去掉
                    if (this.count > 1) {
                        var item = this.itemAt(1);
                        if (item) {
                            item.highlighted = false; // 手动设置 Hover 效果
                        }
                    }
                }

                //timer结束前，如果发生了currentIndex的改变，几乎肯定时弹出时的鼠标恰好hover住了，这种都要过滤掉
                if (!canhighlight) {
                    highlight_firstresult()
                }
            }

            Component.onCompleted: {
                menuKeyFilter.listenTo(newnode_menu)
            }
        }

        Component {
            id: edgescomp
            EdgesContainer { }
        }

        Component.onCompleted: {
            navigator.reset_paths()

            edges = edgescomp.createObject(graphView.containerItem, {
                "graphModel": root.graphModel,
                "graphView": graphView
            });
            if (edges === null) {
                console.error("Failed to create edges object")
            }
        }

        Component {
            id: myNavigator
            Text {
                id: navigatoritem
                color: Qt.rgba(125/255, 125/255, 125/255, 1)
                font.pixelSize: 20
                font.family: "微软雅黑"
                signal graphitemClicked(int idx)
                property int inneridx
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    hoverEnabled: true
                    onClicked: {
                        navigatoritem.graphitemClicked(parent.inneridx)
                    }
                    onEntered: {
                        parent.color = "#0078D4"
                        parent.font.underline = true;
                    }
                    onExited: {
                        parent.color =  Qt.rgba(125/255, 125/255, 125/255, 1)
                        parent.font.underline = false;
                    }
                }
            }
        }

        //子图路径导航栏
        Row {
            id: navigator
            anchors.left: parent.left
            anchors.leftMargin: 15
            anchors.top: parent.top
            anchors.topMargin: 15
            spacing: 8

            function reset_paths() {
                const lst = root.graphModel.path()
                for (let i = navigator.children.length - 1; i >= 0; i--) {
                    let child = navigator.children[i];
                    if (child !== null) {
                        child.destroy(); // 销毁子元素
                    }
                }
                for (var i = 0; i < lst.length; i++)
                {
                    var graphitem = myNavigator.createObject(navigator, { text: lst[i], inneridx: i });
                    graphitem.graphitemClicked.connect(function onGraphitemClicked(inneridx) {
                        //inneridx是点击的文本是第几个，比如 main > Subnet1，如果点击Subnet1则是1，main就是0
                        var path_list = []
                        for (let i = 0; i < navigator.children.length; i++) {
                            let child = navigator.children[i];
                            if (child.text != ">") {
                                path_list.push(child.text);
                                if (inneridx == 0) {
                                    root.navigateRequest(path_list)    //发送信号到上一层(ZenoSubnetsView，跳转图层)
                                    break;
                                }
                                --inneridx;
                            }
                        }
                    })
                    if (i < lst.length - 1) {
                        var seperatoritem = myNavigator.createObject(navigator, { text: ">", inneridx: i });
                    }
                }
            }
        }

        /*    //以下代码不起作用，无法设置widget shortcut
        Shortcut {
            sequence: "Ctrl+S" // 定义快捷键
            context: Qt.WidgetShortcut
            onActivated: {
                console.log("Ctrl+S triggered")
                graphsmanager.saveProject(root.graphModel.name())
            }
        }
        */

        Keys.onPressed: {
            // console.log("Keys.onPressed on ZenoGraphView")
            if (event.key === Qt.Key_P) {
                proppanel.visible = !proppanel.visible
            }
            else if ((event.key === Qt.Key_S) && (event.modifiers & Qt.ControlModifier)) {
                var graphM = root.graphModel;
                var graph_path = graphM.path()
                graphsmanager.saveProject(graph_path[0])
            }
            else if ((event.key === Qt.Key_Z) && (event.modifiers & Qt.ControlModifier)) {
                graphsmanager.undo(root.graphModel.path())
            }
            else if ((event.key === Qt.Key_Y) && (event.modifiers & Qt.ControlModifier)) {
                graphsmanager.redo(root.graphModel.path())
            }
            else if ((event.key === Qt.Key_C) && (event.modifiers & Qt.ControlModifier)) {
                var selNodes = graph.getSelectionNodes()
                graphsmanager.copy(selNodes)
            }
            else if ((event.key === Qt.Key_V) && (event.modifiers & Qt.ControlModifier)) {
                var graphM_path = graphView.graph.model.path()
                var mousepos = graphView.containerItem.mapFromGlobal(MouseUtils.getGlobalMousePosition())
                
                var newnode_names = graphsmanager.paste(mousepos, graphM_path)
                graphView.selectNodes(newnode_names)
            }
            else if (event.key == Qt.Key_Delete) {
                var selected_edges = []
                for (var i = 0; i < edges.children.length; i++) {
                    var edge = edges.children[i]
                    if (edge.state == "select") {
                        selected_edges.push(edge)
                    }
                }
                console.log("selected_edges = " + selected_edges)
                for (var i = 0; i < selected_edges.length; i++) {
                    var selected_edge = selected_edges[i]
                    var inobj = selected_edge.toParam
                    var outobj = selected_edge.fromParam
                    var in_node_name = graphView.graph.getNode(inobj[0]).label
                    var out_node_name = graphView.graph.getNode(outobj[0]).label
                    var in_param = inobj[1]
                    var out_param = outobj[1]
                    var in_key = inobj[3];
                    var out_key = outobj[3]
                    // console.log("in_param = " + in_param + " in_key = " + in_key)
                    // console.log("out_param = " + out_param + " out_key = " + out_key)
                    graphView.graph.model.removeLink(out_node_name, out_param, in_node_name, in_param, out_key, in_key)
                }

            var nodes = []
                var graphM = root.graphModel
                for (var i = 0; i < graph.selectedNodes.length; i++) {
                    var selectNode = graph.selectedNodes.at(i)
                    var clsname = graphM.data(selectNode.index, Model.ROLE_CLASS_NAME);
                    if(!(clsname == "SubInput" || clsname == "SubOutput" )) {
                        nodes.push(selectNode.label)
                    }
                }
                for (var i = 0; i < nodes.length; i++) {
                    graphView.graph.model.removeNode(nodes[i])
                }
            }
        }

        ZPropPanel {
            id: proppanel
            anchors.top: parent.top; anchors.topMargin: 15
            anchors.right: parent.right; anchors.rightMargin: 15
            visible: false

            // 左侧拖动区域
            MouseArea {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: 5
                cursorShape: Qt.SizeHorCursor
                drag.axis: Drag.XAxis
                drag.target: proppanel
                onPositionChanged: {
                    var pt = graphView.mapFromItem(proppanel, Qt.point(mouse.x, 0))
                    var mouseX = pt.x
                    let newWidth = proppanel.width + (proppanel.x - mouseX)
                    var contentW = proppanel.calc_content_width()
                    proppanel.x = mouseX
                    proppanel.width = newWidth
                    //proppanel.width = Math.min(newWidth, contentW) //宽度暂时不做限制
                }
            }

            // 下侧拖动区域
            MouseArea {
                id: down_area
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: 5
                cursorShape: Qt.SizeVerCursor

                onPositionChanged: {
                    var globalpos = down_area.mapToGlobal(mouse.x, mouse.y)
                    var globalY = globalpos.y
                    var viewY = graphView.mapFromGlobal(0, globalY).y
                    let newHeight = viewY - proppanel.y

                    var contentH = proppanel.calc_content_height()
                    proppanel.height = newHeight
                    // proppanel.height = Math.min(newHeight, contentH)
                }
            }
        }

        property var editparamDlg : undefined 
        function showDialog(node) {
            var customuiM = node.params.customUIModel()
            graphsmanager.openCustomUIDialog(customuiM)
        }

        Qan.GraphPreview {
            id: graphPreview
            source: graphView
            viewWindowColor: Qt.rgba(1, 0, 0, 1)//Material.accent
            anchors.right: graphView.right; anchors.bottom: graphView.bottom
            anchors.rightMargin: 8; anchors.bottomMargin: 8
            width: 250
            height: 141
        }
    }  // Qan.GraphView
}