import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import QtQuick.Shapes            1.0
import QtQuick.Controls.Styles   1.4
import Qt.labs.settings 1.1

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import Zeno 1.0 as Zen
import zeno.enum 1.0
import "./view"


Item {
    id: rootgraphsview

    Component {
       id: myTabButton
       CustomTabButton {
           Connections {
                function onCloseTab() {
                    
                }
           }
       }
    }

    function clearOtherTabsButMain() {
        graphs_tabbar.removeItem(1)
        graphs_tabbar.removeItem(2)
        graphs_tabbar.currentIndex = 0
    }

    ToolBar {
        id: zeditortoolbar
        width: parent.width
        property bool view_reentry: false       //用于记录treeview和listview之间互相通知的重入标志
        z: 1000

        background: Rectangle { // 自定义背景颜色
            color: "#1F1F1F"  // 设置背景颜色
        }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 10
            anchors.topMargin: 0
            anchors.bottomMargin: 0
            spacing: 0

            ToolButton {
                id: assets_list
                checkable: true
                checked: false
                
                icon.source: hovered || checked  ? "qrc:/icons/subnet-listview-on.svg" : "qrc:/icons/subnet-listview.svg"

                onClicked: {

                }

                onCheckedChanged: {
                    if (!zeditortoolbar.view_reentry) {
                        zeditortoolbar.view_reentry = true

                        tree_list.checked = false
                        load_plugin.checked = false

                        stack_main_or_asset.visible = checked
                        stack_main_or_asset.currentIndex = 0

                        zeditortoolbar.view_reentry = false
                    }
                }    

                contentItem: Image {
                    id: icon_image1
                    source: parent.icon.source
                    sourceSize.width: 20
                    sourceSize.height: 20
                    smooth: true
                    antialiasing: true
                    //anchors.verticalCenter: parent.verticalCenter
                }
                
                background: Rectangle {
                    x: icon_image1.x
                    y: icon_image1.y
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered || parent.checked ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            ToolButton {
                id: tree_list
                icon.source: hovered || checked  ? "qrc:/icons/nodeEditor_nodeTree_selected.svg" : "qrc:/icons/nodeEditor_nodeTree_unselected.svg"
                checkable: true
                checked: false
                property bool reentry: false

                onCheckedChanged: {
                    if (!zeditortoolbar.view_reentry) {
                        zeditortoolbar.view_reentry = true

                        assets_list.checked = false
                        load_plugin.checked = false

                        stack_main_or_asset.visible = checked
                        stack_main_or_asset.currentIndex = 1
                        
                        zeditortoolbar.view_reentry = false
                    }
                }

                contentItem: Image {
                    id: icon_image2
                    source: parent.icon.source
                    sourceSize.width: 20
                    sourceSize.height: 20
                    smooth: true
                    antialiasing: true
                    //anchors.verticalCenter: parent.verticalCenter
                }

                background: Rectangle {
                    x: icon_image2.x
                    y: icon_image2.y
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered || parent.checked ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            ToolButton {
                id: load_plugin
                icon.source: hovered || checked  ? "qrc:/icons/plugin_selected.svg" : "qrc:/icons/plugin_unselected.svg"
                checkable: true
                checked: false
                property bool reentry: false

                onCheckedChanged: {
                    if (!zeditortoolbar.view_reentry) {
                        zeditortoolbar.view_reentry = true

                        assets_list.checked = false
                        tree_list.checked = false

                        stack_main_or_asset.visible = checked
                        stack_main_or_asset.currentIndex = 2
                        
                        zeditortoolbar.view_reentry = false
                    }
                }

                contentItem: Image {
                    x: 2
                    y: 2
                    id: icon_plugins
                    source: parent.icon.source
                    sourceSize.width: 16
                    sourceSize.height: 16
                    smooth: true
                    antialiasing: true
                }

                background: Rectangle {
                    x: icon_plugins.x - 2
                    y: icon_plugins.y - 2
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            Item { Layout.fillWidth: true }

            Label {
                text: "Auto"
                font.pixelSize: 12
            }
            CheckBox {
                id: cbAlways
                checkState: calcmgr.autoRun ? Qt.Checked : Qt.Unchecked

                onCheckStateChanged: {
                    calcmgr.autoRun = checkState == Qt.Checked
                }
            }

            ToolButton {
                id: run_buttons
                checkable: true
                checked: calcmgr.runStatus == RunStatus.Running
                
                icon.source: checked  ? "qrc:/icons/run_stop.svg" : "qrc:/icons/run_play.svg"

                onClicked: {
                    console.log("Button clicked, checked state:", checked)
                    if (checked) {
                        calcmgr.run()
                    }
                    else{
                        calcmgr.kill()
                    }
                }

                contentItem: Image {
                    id: icon_image3
                    source: parent.icon.source
                    sourceSize.width: 20
                    sourceSize.height: 20
                    smooth: true
                    antialiasing: true
                    //anchors.verticalCenter: parent.verticalCenter
                }

                background: Rectangle {
                    x: icon_image3.x
                    y: icon_image3.y
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            ToolButton {
                id: clean_project
                checkable: false
                
                icon.source: "qrc:/icons/broom_clear_clean_tool.svg"

                onClicked: {
                    calcmgr.clear()
                }

                contentItem: Image {
                    x: 2
                    y: 2
                    id: icon_clean_project
                    source: parent.icon.source
                    sourceSize.width: 16
                    sourceSize.height: 16
                    smooth: true
                    antialiasing: true
                }

                background: Rectangle {
                    x: icon_clean_project.x - 2
                    y: icon_clean_project.y - 2
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            ToolButton {
                id: run_and_clean_buttons
                checkable: true
                checked: calcmgr.runStatus == RunStatus.Running
                
                icon.source: checked ? "qrc:/icons/run_stop.svg" : "qrc:/icons/run_play.svg"

                onClicked: {
                    console.log("Button clicked, checked state:", checked)
                    if (checked) {
                        calcmgr.run_and_clean()
                    }
                    else{
                        calcmgr.kill()
                    }
                }

                contentItem: Image {
                    id: icon_image4
                    source: parent.icon.source
                    sourceSize.width: 20
                    sourceSize.height: 20
                    smooth: true
                    antialiasing: true
                    //anchors.verticalCenter: parent.verticalCenter
                }

                background: Rectangle {
                    x: icon_image4.x
                    y: icon_image4.y
                    width: 20
                    height: 20
                    opacity: enabled ? 1 : 0.3
                    color: parent.hovered ? "#4F5963" : "transparent"
                    border.color: parent.down ? "#17a81a" : "#21be2b"
                    border.width: 0
                    radius: 2
                }
            }

            Item { Layout.fillWidth: true }
        }
    }
    
    SplitView {
        id: mainLayout
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: zeditortoolbar.bottom
        anchors.bottom: parent.bottom

        spacing: 10
        orientation: Qt.Horizontal

        handle: Item {
            implicitWidth: 2

            Rectangle {
                implicitWidth: 2
                anchors.horizontalCenter: parent.horizontalCenter
                height: parent.height
                color: SplitHandle.hovered ? "#00ff00" : "#2B2B2B"
            }
        }

        StackLayout {
            id: stack_main_or_asset
            clip: true
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: 1
            visible: false
            z: 1000

            function activate_asset_tab(assetname) {
                var asset_graph_model = assetsModel.getAssetGraph(assetname)
                for (var i = 0; i < graphs_tabbar.count; i++) {
                    let tab = graphs_tabbar.itemAt(i)
                    if (tab.text == assetname) {
                        //已存在tab，直接激活即可
                        graphs_tabbar.currentIndex = i
                        return;
                    }
                }

                //TODO: 如何为CustomTabButton指定合适的宽度？
                const newtabbutton = myTabButton.createObject(graphs_tabbar, {text: assetname, width: 200, z:1000})
                graphs_tabbar.addItem(newtabbutton)
                newtabbutton.closeTab.connect(function onCloseTab() {
                    for (var i = 0; i < graphs_tabbar.count; i++) {
                        let tab = graphs_tabbar.itemAt(i);
                        if (tab.text == assetname) {
                            //移除当前tab，同时也把stacklayout里面的graphview也一并移除
                            //目前将索引调为一致状态
                            graphs_tabbar.removeItem(i);
                            graphs_stack.children[i].destroy()
                        }
                    }
                })
                                    
                const graphsview_comp = Qt.createComponent("qrc:/ZenoSubnetsView.qml")
                const newgraphview = graphsview_comp.createObject(graphs_stack, {root_graphmodel: asset_graph_model})
                graphs_tabbar.currentIndex = graphs_tabbar.count - 1;
                //TODO: delete and adjust index
            }

            Rectangle {
                id: assets_block
                width: 200
                implicitWidth: 200
                Layout.maximumWidth: 400
                color: "#181818"
                Layout.fillHeight: true

                // 1.定义delegate，内嵌三个Text对象来展示Model定义的ListElement的三个role
                Component {
                    id: assetItemDelegate
                    Item {
                        id: wrapper
                        width: parent.width
                        height: 30

                        //资产列表被点击时，tab新增/跳转:
                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.LeftButton | Qt.RightButton

                            onClicked: {
                                wrapper.ListView.view.currentIndex = index
                                var idx = assetsModel.index(index, 0)
                                var assetname = assetsModel.data(idx)
                                if (mouse.button == Qt.RightButton) {
                                    assetsitem_menu.assetname = assetname
                                    assetsitem_menu.popup(Qt.point(mouse.x + parent.x + 32, mouse.y + parent.y))
                                }
                                else {
                                    stack_main_or_asset.activate_asset_tab(assetname)
                                }
                            }
                        }

                        RowLayout {
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.leftMargin: 10
                            spacing: 8

                            Text {
                                id: lbl_assetname
                                text: classname
                                // 是否是当前条目
                                color: wrapper.ListView.isCurrentItem ? "white" : "#C3D2DF"
                                font.pixelSize: 22
                                Layout.preferredWidth: 120
                            }
                        }
                    }
                }

                Menu {
                    id: assetMenu
                    property var node: null

                    MenuItem {
                        text: "创建Zeno资产(zda)"
                        onTriggered: {
                            assetMenu.close()
                            Qt.callLater(() => {   // 再延迟打开对话框
                                graphsmanager.createAssetDialog()
                            });
                        }
                    }

                    MenuItem {
                        text: "打开资产文件"
                        onTriggered: {
                            assetMenu.close()
                            Qt.callLater(() => {   // 再延迟打开对话框
                                graphsmanager.loadAssetDialog()
                            });
                        }
                    }
                }

                Menu {
                    id: assetsitem_menu
                    property var assetname: ""

                    MenuItem {
                        text: "Custom UI"
                        onTriggered: {
                            assetsitem_menu.close()
                            Qt.callLater(() => {   // 再延迟打开对话框
                                graphsmanager.onAssetsCustomUIDialog(assetsitem_menu.assetname)
                            });
                        }
                    }
                    MenuItem {
                        text: "Open Container Folder"
                        onTriggered: {}
                    }
                } 

                ColumnLayout {
                    anchors.fill: parent
                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            text: "Assets:"
                            font.pixelSize: 16
                            color: "grey"
                        }

                        Item {
                            Layout.fillWidth: true
                        }

                        ToolButton {
                            // id: plugins_more
                            checkable: false
                            width: 16
                            height: 16
                            
                            icon.source: hovered ? "qrc:/icons/add-on.svg" : "qrc:/icons/add.svg"

                            onClicked: {
                                assetMenu.popup()
                            }
                
                            contentItem: Image {
                                x: 2
                                y: 2
                                source: parent.icon.source
                                sourceSize.width: 16
                                sourceSize.height: 16
                                smooth: true
                                antialiasing: true
                            }

                            background: Rectangle {
                                width: 20
                                height: 20
                                anchors.fill: parent
                                color: "transparent"  // 不改变背景，只保证 hover 区域
                            }
                        }
                    }

                    Rectangle {
                        height: 1
                        Layout.fillWidth: true
                        color: "grey"
                    }

                    ListView {
                        id: listView
                        Layout.fillHeight: true
                        Layout.fillWidth: true

                        // 使用先前设置的delegate
                        delegate: assetItemDelegate
                        
                        // 3.ListModel专门定义列表数据的，它内部维护一个 ListElement 的列表。
                        model: assetsModel

                        // 背景高亮
                        focus: true
                        highlight: Rectangle{
                            color: "#3D3D3D"
                        }
                    }
                }
            }

            Rectangle {
                SplitView.preferredWidth: 340
                Layout.fillHeight: true
                Layout.fillWidth: true
                color: "#181818"

                TreeView {
                    id: styledTreeView

                    Layout.fillHeight: true

                    model: treeModel
                    selectionEnabled: true
                    hoverEnabled: true

                    color: "#AAAACC"
                    handleColor: "#B0CCCC"
                    hoverColor: "#2A2D2E"
                    selectedColor: "#37373D"
                    selectedItemColor: "white"
                    handleStyle: TreeView.Handle.TriangleOutline
                    rowHeight: 40
                    rowPadding: 30
                    rowSpacing: 12
                    font.pixelSize: 20

                    onCurrentIndexChanged: {
                        //main图现在都是默认打开的。
                        if (currentIndex == null) {
                            return;
                        }
                        var graphM = model.graph(currentIndex)
                        if (graphM == null) {
                            console.log("warning: graphM is null");
                            return;
                        }

                        var path_list = graphM.path()
                        graphs_tabbar.currentIndex = 0;     //切换到主图的view
                        stack_main_graphview.jumpTo(path_list);

                        //var ident = model.ident(currentIndex)
                        //var owner = graphM.owner()
                        //console.log("ident: " + ident)
                        //console.log("owner: " + owner)
                        //tabView打开标签为owner的图，并且把焦点focus在ident上。
                        //app.tab.activatePage(owner, graphM)
                    }
                    onCurrentDataChanged: {
                        //console.log("current data is " + currentData)
                    }
                    onCurrentItemChanged: {
                        //console.log("current item is " + currentItem)
                    }
                }
            }

            Item {
                id: plugins_list

                // implicitWidth: 200
                // Layout.maximumWidth: 400
                Layout.fillHeight: true
                Layout.fillWidth: true

                Rectangle {
                    anchors.fill: parent
                    color: "#181818"
                }

                Menu {
                    id: pluginitem_menu
                    MenuItem {
                        text: "Remove Plugin"
                        onTriggered: {
                            pluginsModel.removePlugin(listView_plugins.currentIndex);
                        }
                    }
                    MenuItem {
                        text: "Open Container Folder"
                        onTriggered: {

                        }
                    }
                }                

                Component {
                    id: pluginItemDelegate
                    Item {
                        id: _item
                        width: parent.width
                        height: 30

                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.LeftButton | Qt.RightButton

                            onClicked: {
                                _item.ListView.view.currentIndex = index
                                if (mouse.button == Qt.RightButton) {
                                    pluginitem_menu.popup(Qt.point(mouse.x + parent.x + 32, mouse.y + parent.y));
                                }
                            }
                        }

                        RowLayout {
                            anchors.left: parent.left
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.leftMargin: 10
                            spacing: 8

                            Text {
                                id: lbl_plugin
                                text: name
                                // 是否是当前条目
                                color: _item.ListView.isCurrentItem ? "white" : "#C3D2DF"
                                font.pixelSize: 14
                                Layout.preferredWidth: 120
                            }
                        }
                    }
                }

                ColumnLayout {
                    anchors.fill: parent
                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            text: "Loaded Plugins:"
                            font.pixelSize: 16
                            color: "grey"
                        }

                        Item {
                            Layout.fillWidth: true
                        }

                        ToolButton {
                            // id: plugins_more
                            checkable: false
                            width: 16
                            height: 16
                            
                            icon.source: hovered ? "qrc:/icons/add-on.svg" : "qrc:/icons/add.svg"

                            onClicked: {
                                graphsmanager.addPlugin()
                            }
                
                            contentItem: Image {
                                x: 2
                                y: 2
                                source: parent.icon.source
                                sourceSize.width: 16
                                sourceSize.height: 16
                                smooth: true
                                antialiasing: true
                            }

                            background: Rectangle {
                                width: 20
                                height: 20
                                anchors.fill: parent
                                color: "transparent"  // 不改变背景，只保证 hover 区域
                            }
                        }
                    }

                    Rectangle {
                        height: 1
                        Layout.fillWidth: true
                        color: "grey"
                    }

                    ListView {
                        id: listView_plugins
                        Layout.fillHeight: true
                        Layout.fillWidth: true

                        // 使用先前设置的delegate
                        delegate: pluginItemDelegate
                            
                        // 3.ListModel专门定义列表数据的，它内部维护一个 ListElement 的列表。
                        model: pluginsModel

                        // 背景高亮
                        focus: true
                        highlight: Rectangle{
                            color: "#3D3D3D"
                        }
                    }
                }
            }
        }

        Item {
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            Layout.fillWidth: true;
            Layout.fillHeight: true

            TabBar {
                id: graphs_tabbar
                z: 1000
                anchors.left: parent.left
                anchors.right: parent.right

                background: Rectangle {
                    color: Qt.rgba(59./255, 59./255, 59./255,1.0) // 设置背景颜色
                }

                CustomTabButton {
                    text: qsTr("main")
                    width: 100
                    z: 1000
                    closable: false
                }

                /*
                CustomTabButton {
                    text: qsTr("Asset1")
                    width: implicitWidth
                }
                CustomTabButton {
                    text: qsTr("Asset2")
                    width: implicitWidth
                }
                */
            }

            StackLayout {
                id: graphs_stack
                width: parent.width
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: graphs_tabbar.bottom
                anchors.bottom: parent.bottom
                currentIndex: graphs_tabbar.currentIndex

                ZenoSubnetsView {
                    id: stack_main_graphview
                    root_graphmodel: nodesModel
                }

                function stepIntoSubnet(path_list) {
                    graphs_stack.children[graphs_stack.currentIndex].jumpTo(path_list)
                }

                function jumpToAsset(asset_name) {
                    stack_main_or_asset.activate_asset_tab(asset_name)
                }

                /*
                Item {
                    Loader { anchors.fill: parent; source: "qrc:/zenographview.qml";  }
                }
                */
            }
        }
    }
}