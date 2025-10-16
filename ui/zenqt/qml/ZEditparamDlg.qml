import QtQuick                   2.12
import QtQuick.Controls          2.15
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3
import zeno.enum 1.0
import "./controls"
import "./view"
import QtQuick.Window 2.15

// Dialog {
//     id: dialog
//     title: "参数设置"
//     width: 800
//     height: 600
//     modal: true

//     property var node: undefined

//     contentItem: Row {  // 水平布局分为三部分
//         anchors.fill: parent

//         // 左侧列表视图
//         ListView {
//             id: leftListView
//             width: parent.width * 0.2
//             height: parent.height
//             model: ListModel {
//                 ListElement { name: "Tab" }
//                 ListElement { name: "Group" }
//                 ListElement { name: "Integer" }
//                 ListElement { name: "Float" }
//             }
//             delegate: Item {
//                 width: parent.width
//                 height: 30
//                 Text {
//                     anchors.centerIn: parent
//                     text: model.name
//                 }
//             }
//         }

//         TreeView {
//             id: styledTreeView

//             Layout.fillHeight: true

//             model: ListModel {
//                 ListElement { name: "Tab" }
//                 ListElement { name: "Group" }
//                 ListElement { name: "Integer" }
//                 ListElement { name: "Float" }
//             }
//             selectionEnabled: true
//             hoverEnabled: true

//             color: "#AAAACC"
//             handleColor: "#B0CCCC"
//             hoverColor: "#2A2D2E"
//             selectedColor: "#37373D"
//             selectedItemColor: "white"
//             handleStyle: TreeView.Handle.TriangleOutline
//             rowHeight: 40
//             rowPadding: 30
//             rowSpacing: 12
//             font.pixelSize: 20

//             onCurrentIndexChanged: {
//                 //main图现在都是默认打开的。
//                 if (currentIndex == null) {
//                     return;
//                 }
//                 var graphM = model.graph(currentIndex)
//                 if (graphM == null) {
//                     console.log("warning: graphM is null");
//                     return;
//                 }

//                 var path_list = graphM.path()
//                 graphs_tabbar.currentIndex = 0;     //切换到主图的view
//                 stack_main_graphview.jumpTo(path_list);

//                 //var ident = model.ident(currentIndex)
//                 //var owner = graphM.owner()
//                 //console.log("ident: " + ident)
//                 //console.log("owner: " + owner)
//                 //tabView打开标签为owner的图，并且把焦点focus在ident上。
//                 //app.tab.activatePage(owner, graphM)
//             }
//             onCurrentDataChanged: {
//                 //console.log("current data is " + currentData)
//             }
//             onCurrentItemChanged: {
//                 //console.log("current item is " + currentItem)
//             }
//         }

//         // // 中间部分
//         // Column {
//         //     width: parent.width * 0.5
//         //     height: parent.height
//         //     spacing: 10

//             // // 树形视图
//             // TreeView {
//             //     anchors.fill: parent
//             //     model: ListModel {
//             //         ListElement {
//             //             name: "Item 1"
//             //         }
//             //         ListElement {
//             //             name: "Item 2"
//             //         }
//             //     }

//             //     TableViewColumn {
//             //         role: "name"
//             //         title: "Name"
//             //         width: 300
//             //     }
//             // }

//             // // 下方三个列表视图
//             // ListView {
//             //     id: listView1
//             //     height: 80
//             //     model: ListModel { ListElement { name: "output1" } }
//             //     delegate: Text { text: model.name }
//             // }

//             // ListView {
//             //     id: listView2
//             //     height: 80
//             //     model: ListModel { ListElement { name: "objInput1" } }
//             //     delegate: Text { text: model.name }
//             // }

//             // ListView {
//             //     id: listView3
//             //     height: 80
//             //     model: ListModel { ListElement { name: "objOutput1" } }
//             //     delegate: Text { text: model.name }
//             // }
//         // }

//         // 右侧部分
//         Column {
//             width: parent.width * 0.3
//             spacing: 10

//             TextField {
//                 id: inputName
//                 placeholderText: "名称"
//             }

//             TextField {
//                 id: inputLabel
//                 placeholderText: "标签"
//             }

//             ComboBox {
//                 id: socketProperty
//                 model: ["Property 1", "Property 2", "Property 3"]
//                 currentIndex: 0
//             }
//         }
//     }

//     // 标准对话框按钮
//     footer: Row {
//         spacing: 10
//         Button {
//             text: "应用"
//             onClicked: console.log("应用点击")
//         }
//         Button {
//             text: "取消"
//             onClicked: dialog.close()
//         }
//     }
// }






// 自定义对话框
Window {
    id: dialog
    width: 800
    height: 600
    visible: false
    title: "对话框"
    flags: Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint // 设置窗口类型为对话框，带标题栏和关闭按钮
    modality: Qt.ApplicationModal  // 将窗口设置为模态
    property var node: undefined

    property var controlItemM : undefined
    property var tabelM : undefined
    property var primOutputM : undefined
    property var objInputM : undefined
    property var objOutputM : undefined


    MouseArea {
        anchors.fill: parent
        // 对话框内容
        Rectangle {
            id: rootRec
            color: "#1f1f1f"
            width: parent.width
            height: parent.height
            // color: "lightgray"

            // Text {
            //     text: "这是一个带标题栏的可移动对话框。\n您确定要继续吗？"
            //     anchors.centerIn: parent
            //     horizontalAlignment: Text.AlignHCenter
            //     wrapMode: Text.WordWrap
            // }
            property int currentTabRow: -1
            property int currentGroupRow: -1
            property int currentParamRow: -1

            RowLayout {  // 水平布局分为三部分
                id: rootLayout
                // anchors.fill: parent
                // anchors.top: parent.top
                width: parent.width
                height: parent.height * 14 / 15
            
                // 左侧列表视图
                Rectangle {
                    color: "#22252C"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 3

                    ListView {
                        id: leftListView
                        anchors.fill: parent
                        property int currentIdx : 0

                        model: controlItemM
                        delegate: Rectangle {
                            width: ListView.view.width
                            height: 20
                            color: leftListView.currentIdx === index ? "lightblue" : "transparent"

                            Row {
                                anchors.fill: parent
                                spacing: 10 // 图标和文本之间的间距
                                leftPadding: 15 // 左侧留白
                                Image {
                                    visible: model.ctrlItemICon !== ""
                                    source: model.ctrlItemICon
                                    width: parent.height
                                    height: parent.height
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    text: model.ctrlItemName
                                    color: leftListView.currentIdx === index ? "black" : "lightgray"
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    leftListView.currentIdx = index  // 设置当前选中的条目
                                }
                            }
                        }
                    }
                }

                // 中间部分
                Item {//使用item做一层隔离，否则会报：QML QQuickLayoutAttached: Binding loop detected for property循环绑定警告
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 3

                    ColumnLayout {
                        // 下方三个列表视图
                        anchors.fill: parent

                        Rectangle {
                            id: midRect1
                            color: "transparent"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: parent.height * 2 / 5

                            RowLayout {
                                anchors.fill: parent
                                Button {
                                    implicitWidth: 30
                                    implicitHeight: 30
                                    padding: 0
                                    text: "add"
                                    font.pointSize: 6
                                    onClicked: {
                                        var tabmodel = dialog.tabelM
                                        var modelRef = undefined
                                        var curRow = -1
                                        var newname = ""
                                        if(leftListView.currentIdx == 2){return}//obj类型return
                                        if(leftListView.currentIdx === 0){
                                            modelRef = tabmodel
                                            newname = getNewName()
                                            if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1){
                                                curRow = rootRec.currentTabRow + 1
                                            }else if(rootRec.currentTabRow === -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1){
                                                curRow = modelRef.rowCount()
                                            }else{
                                                return
                                            }
                                            modelRef.insertRow(curRow, newname)
                                        }else if(leftListView.currentIdx === 1){
                                            if(rootRec.currentTabRow === -1){return}
                                            modelRef = tabmodel.data(tabmodel.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
                                            newname = getNewName()
                                            if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1) {
                                                curRow = modelRef.rowCount()
                                            }else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow === -1){
                                                curRow = rootRec.currentGroupRow + 1
                                            }else{
                                                return
                                            }
                                            modelRef.insertRow(curRow, newname)
                                        }else{
                                            if(rootRec.currentTabRow === -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1 ||
                                            rootRec.currentTabRow !== -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1
                                            )
                                            {return}
                                            var groupmodel = tabmodel.data(tabmodel.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
                                            modelRef =  groupmodel.data(groupmodel.index(rootRec.currentGroupRow, 0), CustomuiModelType.PrimModel)
                                            newname = getNewName()
                                            if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow === -1) {
                                                curRow = modelRef.rowCount()
                                            }else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow !== -1){
                                                curRow = rootRec.currentParamRow + 1
                                            }else{
                                                return
                                            }
                                            modelRef.insertRow(curRow, newname, leftListView.currentIdx)
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    color: "#22252C"
                                    Loader {
                                        id: treeviewLoader
                                        anchors.fill: parent
                                        // Layout.fillWidth: true
                                        // Layout.fillHeight: true
                                        sourceComponent: {
                                            return treeview
                                        }
                                    }
                                }
                            }
                            Keys.onDeletePressed : {
                                deleteItem()
                            }
                        }

                        Rectangle {
                            color: "transparent"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: parent.height / 5
                            RowLayout {
                                anchors.fill: parent
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width / 10
                                }    
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width * 9 / 10

                                    ColumnLayout {
                                        id: primoutColumn
                                        anchors.fill: parent
                                        property var primOutputModel : dialog.primOutputM

                                        Item {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 25

                                            RowLayout{
                                                anchors.fill: parent
                                                Item {
                                                    Layout.alignment: Qt.AlignLeft | Qt.AlignVCenter
                                                    Layout.fillHeight: true
                                                    Layout.fillWidth: true
                                                    Text {
                                                        color: "lightgray"
                                                        // anchors.verticalCenter: parent.verticalCenter
                                                        font.bold: true
                                                        text: "OutputPrims"
                                                    }
                                                }
                                                Button {
                                                    Layout.alignment: Qt.AlignTop | Qt.AlignRight
                                                    implicitWidth: 30
                                                    implicitHeight: 25
                                                    padding: 0
                                                    text: "add"
                                                    font.pointSize: 6
                                                    onClicked: {
                                                        if(listView1.currentIdx === -1){
                                                            primoutColumn.primOutputModel.insertRow(primoutColumn.primOutputModel.rowCount(), getNewName())
                                                        }else{
                                                            primoutColumn.primOutputModel.insertRow(listView1.currentIdx + 1, getNewName())
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            color: "#22252C"
                                            ListView {
                                                id: listView1
                                                anchors.fill: parent
                                                anchors.leftMargin: 20
                                                // Layout.fillWidth: true
                                                // Layout.fillHeight: true
                                                property int currentIdx : -1
                                                clip: true
                                                // model: ListModel { ListElement { name: "output1" } }
                                                model: primoutColumn.primOutputModel
                                                delegate: Rectangle {
                                                    width: ListView.view.width
                                                    height: 15
                                                    color: listView1.currentIdx === index ? "lightblue" : "transparent"
                                                    // Text { text: model.paramname }

                                                    Loader {
                                                        id: lstV1TextEditLoader
                                                        anchors.fill: parent
                                                        sourceComponent: textEditComp
                                                        Binding {
                                                            target: lstV1TextEditLoader.item
                                                            property: "textContent"
                                                            value: model.name
                                                        }
                                                        Binding {
                                                            target: lstV1TextEditLoader.item
                                                            property: "textClr"
                                                            value: listView1.currentIdx === index ? "black" : "lightgray"
                                                        }
                                                    }
                                                    Connections {
                                                        target: lstV1TextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                                        function onEditfinish() {
                                                            primoutColumn.primOutputModel.setData(primoutColumn.primOutputModel.index(listView1.currentIdx, 0), lstV1TextEditLoader.item.textContent, Model.ROLE_PARAM_NAME)
                                                        }
                                                    }
                                                    MouseArea {
                                                        anchors.fill: parent
                                                        enabled: !lstV1TextEditLoader.item.editMode
                                                        onClicked: {
                                                            listView1.currentIdx = index  // 设置当前选中的条目
                                                            listView1.forceActiveFocus()
                                                        }
                                                        onDoubleClicked: {
                                                            lstV1TextEditLoader.item.editMode = true;
                                                            lstV1TextEditLoader.item.inputfield.forceActiveFocus();
                                                        }
                                                    }
                                                }
                                                Keys.onDeletePressed: {
                                                    var name = primoutColumn.primOutputModel.data(primoutColumn.primOutputModel.index(listView1.currentIdx, 0), Model.ROLE_PARAM_NAME)
                                                    if(name == "data_output"){
                                                        return
                                                    }
                                                    primoutColumn.primOutputModel.removeRow(listView1.currentIdx)
                                                    if(primoutColumn.primOutputModel.rowCount() === 0 ){
                                                        listView1.currentIdx = -1
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }  
                            }
                        }

                        Rectangle {
                            color: "transparent"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: parent.height / 5
                            RowLayout {
                                anchors.fill: parent
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width / 10
                                }    
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width * 9 / 10

                                    ColumnLayout {
                                        id: objinColumn
                                        anchors.fill: parent
                                        property var objInputModel : dialog.objInputM

                                        Item {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 25

                                            RowLayout{
                                                anchors.fill: parent
                                                Item {
                                                    Layout.alignment: Qt.AlignLeft | Qt.AlignVCenter
                                                    Layout.fillHeight: true
                                                    Layout.fillWidth: true
                                                    Text {
                                                        color: "lightgray"
                                                        // anchors.verticalCenter: parent.verticalCenter
                                                        font.bold: true
                                                        text: "InputObj"
                                                    }
                                                }
                                                Button {
                                                    Layout.alignment: Qt.AlignTop | Qt.AlignRight
                                                    implicitWidth: 30
                                                    implicitHeight: 25
                                                    padding: 0
                                                    text: "add"
                                                    font.pointSize: 6
                                                    onClicked: {
                                                        if(listView2.currentIdx === -1){
                                                            objinColumn.objInputModel.insertRow(objinColumn.objInputModel.rowCount(), getNewName())
                                                        }else{
                                                            objinColumn.objInputModel.insertRow(listView2.currentIdx + 1, getNewName())
                                                        }
                                                    }
                                                }
                                                Button {
                                                    Layout.alignment: Qt.AlignTop | Qt.AlignRight
                                                    implicitWidth: 30
                                                    implicitHeight: 25
                                                    padding: 0
                                                    text: "remove"
                                                    font.pointSize: 6
                                                    onClicked: {
                                                        objinColumn.objInputModel.removeRow(listView2.currentIdx)
                                                    }
                                                }
                                            }
                                        }
                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            color: "#22252C"
                                            ListView {
                                                id: listView2
                                                anchors.fill: parent
                                                anchors.leftMargin: 20
                                                // Layout.fillWidth: true
                                                // Layout.fillHeight: true
                                                property int currentIdx : -1
                                                clip: true
                                                model: objinColumn.objInputModel
                                                delegate: Rectangle {
                                                    width: ListView.view.width
                                                    height: 15
                                                    color: listView2.currentIdx === index ? "lightblue" : "transparent"
                                                    // Text { text: model.paramname }
                                                    Loader {
                                                        id: lstV2TextEditLoader
                                                        anchors.fill: parent
                                                        sourceComponent: textEditComp
                                                        Binding {
                                                            target: lstV2TextEditLoader.item
                                                            property: "textContent"
                                                            value: model.paramname
                                                        }
                                                        Binding {
                                                            target: lstV2TextEditLoader.item
                                                            property: "textClr"
                                                            value: listView2.currentIdx === index ? "black" : "lightgray"
                                                        }
                                                    }
                                                    Connections {
                                                        target: lstV2TextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                                        function onEditfinish() {
                                                            objinColumn.objInputModel.setData(objinColumn.objInputModel.index(listView2.currentIdx, 0), lstV2TextEditLoader.item.textContent, Model.ROLE_PARAM_NAME)
                                                        }
                                                    }

                                                    MouseArea {
                                                        anchors.fill: parent
                                                        enabled: !lstV2TextEditLoader.item.editMode
                                                        onClicked: {
                                                            listView2.currentIdx = index  // 设置当前选中的条目
                                                            listView2.forceActiveFocus()
                                                        }
                                                        onDoubleClicked: {
                                                            lstV2TextEditLoader.item.editMode = true;
                                                            lstV2TextEditLoader.item.inputfield.forceActiveFocus();
                                                        }
                                                    }
                                                }
                                                Keys.onDeletePressed: {
                                                    var name = objinColumn.objInputModel.data(objinColumn.objInputModel.index(listView2.currentIdx, 0), Model.ROLE_PARAM_NAME)
                                                    if(name == "Input"){
                                                        return
                                                    }
                                                    objinColumn.objInputModel.removeRow(listView2.currentIdx)
                                                    if(objinColumn.objInputModel.rowCount() === 0 ){
                                                        listView2.currentIdx = -1
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Rectangle {
                            color: "transparent"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.preferredHeight: parent.height / 5
                            RowLayout {
                                anchors.fill: parent
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width / 10
                                }    
                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.preferredWidth: parent.width * 9 / 10

                                    ColumnLayout {
                                        id: objoutColumn
                                        anchors.fill: parent
                                        property var objOutputModel : dialog.objOutputM

                                        Item {
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 25

                                            RowLayout{
                                                anchors.fill: parent
                                                Item {
                                                    Layout.alignment: Qt.AlignLeft | Qt.AlignVCenter
                                                    Layout.fillHeight: true
                                                    Layout.fillWidth: true
                                                    Text {
                                                        color: "lightgray"
                                                        // anchors.verticalCenter: parent.verticalCenter
                                                        font.bold: true
                                                        text: "OutputObj"
                                                    }
                                                }
                                                Button {
                                                    Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                                                    implicitWidth: 30
                                                    implicitHeight: 25
                                                    padding: 0
                                                    text: "add"
                                                    font.pointSize: 6
                                                    onClicked: {
                                                        if(listView3.currentIdx === -1){
                                                            objoutColumn.objOutputModel.insertRow(objoutColumn.objOutputModel.rowCount(), getNewName())
                                                        }else{
                                                            objoutColumn.objOutputModel.insertRow(listView3.currentIdx + 1, getNewName())
                                                        }
                                                    }
                                                }
                                                Button {
                                                    Layout.alignment: Qt.AlignTop | Qt.AlignRight
                                                    implicitWidth: 40
                                                    implicitHeight: 25
                                                    padding: 0
                                                    text: "remove"
                                                    font.pointSize: 6
                                                    onClicked: {
                                                        objoutColumn.objOutputModel.removeRow(listView3.currentIdx)
                                                    }
                                                }
                                            }
                                        }
                                        Rectangle {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            color: "#22252C"
                                            ListView {
                                                id: listView3
                                                anchors.fill: parent
                                                anchors.leftMargin: 20
                                                // Layout.fillWidth: true
                                                // Layout.fillHeight: true
                                                property int currentIdx : -1
                                                clip: true
                                                model: objoutColumn.objOutputModel
                                                delegate: Rectangle {
                                                    width: ListView.view.width
                                                    height: 15
                                                    color: listView3.currentIdx === index ? "lightblue" : "transparent"
                                                    // Text { text: model.paramname }
                                                    Loader {
                                                        id: lstV3TextEditLoader
                                                        anchors.fill: parent
                                                        sourceComponent: textEditComp
                                                        Binding {
                                                            target: lstV3TextEditLoader.item
                                                            property: "textContent"
                                                            value: model.paramname
                                                        }
                                                        Binding {
                                                            target: lstV3TextEditLoader.item
                                                            property: "textClr"
                                                            value: listView3.currentIdx === index ? "black" : "lightgray"
                                                        }
                                                    }
                                                    Connections {
                                                        target: lstV3TextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                                        function onEditfinish() {
                                                            objoutColumn.objOutputModel.setData(objoutColumn.objOutputModel.index(listView3.currentIdx, 0), lstV3TextEditLoader.item.textContent, Model.ROLE_PARAM_NAME)
                                                        }
                                                    }

                                                    MouseArea {
                                                        anchors.fill: parent
                                                        enabled: !lstV3TextEditLoader.item.editMode
                                                        onClicked: {
                                                            listView3.currentIdx = index  // 设置当前选中的条目
                                                            listView3.forceActiveFocus()
                                                        }
                                                        onDoubleClicked: {
                                                            lstV3TextEditLoader.item.editMode = true;
                                                            lstV3TextEditLoader.item.inputfield.forceActiveFocus();
                                                        }
                                                    }
                                                }
                                                Keys.onDeletePressed: {
                                                    var name = objoutColumn.objOutputModel.data(objoutColumn.objOutputModel.index(listView3.currentIdx, 0), Model.ROLE_PARAM_NAME)
                                                    if(name == "Output"){
                                                        return
                                                    }
                                                    objoutColumn.objOutputModel.removeRow(listView3.currentIdx)
                                                     if(objoutColumn.objOutputModel.rowCount() === 0 ){
                                                        listView3.currentIdx = -1
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    }
                }

                RowLayout{
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width / 3

                    Item {
                        implicitWidth: 30
                        implicitHeight: 30
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#22252C"
                        // 右侧部分
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 10

                            TextField {
                                id: inputName
                                text: "AAA"
                                color: "lightgray"
                                placeholderText: "名称"
                            }

                            TextField {
                                id: inputLabel
                                text: "label"
                                color: "lightgray"
                                placeholderText: "标签"
                            }

                            ComboBox {
                                id: socketProperty
                                model: ["Property 1", "Property 2", "Property 3"]
                                currentIndex: 0
                            }
                        }
                    }
                }
            }

            Item {
                width: parent.width
                height: parent.height / 15
                anchors.top: rootLayout.bottom
                anchors.horizontalCenter: rootLayout.horizontalCenter
                // anchors.bottom: parent.bottom
                // anchors.bottomMargin: 5

                RowLayout {
                    anchors.fill: parent

                    Item {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        Layout.preferredWidth : parent.width * 5 / 6
                    }
                    Item {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        Layout.preferredWidth : parent.width / 6
                        Layout.rightMargin: 20
                        Layout.alignment: Qt.AlignVCenter
                        RowLayout {
                            Layout.fillHeight: true
                            Layout.fillWidth: true
                            Button {
                                // anchors.left: parent.left
                                implicitWidth: 60
                                implicitHeight: 40
                                text: "确定"
                                onClicked: {
                                    node.params.applyParamsByEditparamDlg(dialog.tabelM.uimodel());
                                    dialog.visible = false
                                }
                            }
                            Button {
                                // anchors.right: parent.right
                                implicitWidth: 60
                                implicitHeight: 40
                                text: "取消"
                                onClicked: {
                                    node.params.cancleEditCustomUIModelCloned()
                                    dialog.visible = false
                                }
                            }
                        }
                    }
                }

                // Button {
                //     text: "取消"
                //     onClicked: {
                //         console.log("点击了取消")
                //         dialog.visible = false
                //     }
                // }
            }

        }
        //为了让textfield失去焦点
        onClicked: {
            // console.log(textField.textContent)
            forceActiveFocus()
        }
    }

    function openDialog() {
        dialog.visible = true
    }
    function getNewName() {
        var nameSet = {}
        var subfix = 0
        var newname = "newitem" + subfix

        for(var i = 0; i < dialog.tabelM.rowCount(); i++) {
            nameSet[dialog.tabelM.data(dialog.tabelM.index(i, 0), Model.ROLE_PARAM_NAME)] = true
            var groupmodel = dialog.tabelM.data(dialog.tabelM.index(i, 0), CustomuiModelType.GroupModel)
            for(var j = 0; j < groupmodel.rowCount(); ++j) {
                nameSet[groupmodel.data(groupmodel.index(j, 0), Qt.DisplayRole)] = true
                var parammodel = groupmodel.data(groupmodel.index(j, 0), CustomuiModelType.PrimModel)
                for(var k = 0; k < parammodel.rowCount(); ++k) {
                    nameSet[parammodel.data(parammodel.index(k, 0), Model.ROLE_PARAM_NAME)] = true
                }
            }
        }
        for(var i = 0; i < dialog.primOutputM.rowCount(); i++){
            nameSet[dialog.primOutputM.data(dialog.primOutputM.index(i, 0), Model.ROLE_PARAM_NAME)] = true
        }
        for(var i = 0; i < dialog.objInputM.rowCount(); i++){
            nameSet[dialog.objInputM.data(dialog.objInputM.index(i, 0), Model.ROLE_PARAM_NAME)] = true
        }
        for(var i = 0; i < dialog.objOutputM.rowCount(); i++){
            nameSet[dialog.objOutputM.data(dialog.objOutputM.index(i, 0), Model.ROLE_PARAM_NAME)] = true
        }
        while (nameSet[newname] !== undefined) {
            subfix++
            newname = "newitem" + subfix
        }
        return newname
    }

    function deleteItem(){
        var modelRef
        var curRow = 0
        if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1) {//删tab
            var tabname =  dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0),  Model.ROLE_PARAM_NAME)
            if(tabname == "Tab1"){
                return false
            }
            modelRef = dialog.tabelM
            curRow = rootRec.currentTabRow
        } else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow === -1) {//删group
            var groupmodel = dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
            var groupname =  groupmodel.data(groupmodel.index(rootRec.currentGroupRow, 0),  Qt.DisplayRole)
            if(groupname == "Group1"){
                return false
            }
            modelRef = groupmodel
            curRow = rootRec.currentGroupRow
        } else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow !== -1) {//删param
            var groupmodel = dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
            modelRef = groupmodel.data(groupmodel.index(rootRec.currentGroupRow, 0), CustomuiModelType.PrimModel)
            var paramname = modelRef.data(modelRef.index(rootRec.currentParamRow, 0), Model.ROLE_PARAM_NAME)
            if(paramname == "data_input"){
                return false
            }
            curRow = rootRec.currentParamRow
        } else {
            console.log("error index")
            return false
        }
        modelRef.removeRow(curRow)
        return true
    }
    Component {
        id: textEditComp
        Item {
            anchors.fill: parent
            property string textContent
            property bool editMode: false
            property alias inputfield : inputField
            property alias textClr: inputField.color
            signal editfinish()

            TextInput {
                id: inputField
                color: "lightgray"
                anchors.fill: parent
                text: textContent
                verticalAlignment: Text.AlignVCenter
                selectByMouse: true // 启用鼠标选择文本功能
                clip: true
                focus: editMode // 聚焦逻辑绑定到 editMode
                Keys.onReturnPressed: {
                    textContent = text;
                    editMode = false;   // 取消编辑并切回文本显示
                    editfinish()
                }
                Keys.onEscapePressed: {
                    textContent = text;
                    editMode = false;   // 取消编辑并切回文本显示
                    editfinish()
                }
                onEditingFinished: {
                    textContent = text;
                    editMode = false;
                    editfinish()
                }
            }
        }
    }

    Component {
        id: treeview
        Item {
            id: treeviewItem
            anchors.fill: parent

            Menu {
                id: contextMenu
                property var modelRef
                property var curRow : undefined
                MenuItem {
                    text: "新增条目"
                    onTriggered: {
                        if(leftListView.currentIdx == 2){return}//obj类型return
                        if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1) {
                            contextMenu.modelRef = dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
                            contextMenu.curRow = contextMenu.modelRef.rowCount()
                        } else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow === -1) {
                            var groupmodel = dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
                            contextMenu.modelRef = groupmodel.data(groupmodel.index(rootRec.currentGroupRow, 0), CustomuiModelType.PrimModel)
                            contextMenu.curRow = contextMenu.modelRef.rowCount()
                        } else if(rootRec.currentTabRow !== -1 && rootRec.currentGroupRow !== -1 && rootRec.currentParamRow !== -1) {
                            var groupmodel = dialog.tabelM.data(dialog.tabelM.index(rootRec.currentTabRow, 0), CustomuiModelType.GroupModel)
                            contextMenu.modelRef = groupmodel.data(groupmodel.index(rootRec.currentGroupRow, 0), CustomuiModelType.PrimModel)
                            contextMenu.curRow = rootRec.currentParamRow + 1
                            var newname = getNewName()
                            contextMenu.modelRef.insertRow(contextMenu.curRow, newname, leftListView.currentIdx)
                            return
                        } else {
                            console.log("error index")
                            return
                        }
                        var newname = getNewName()
                        contextMenu.modelRef.insertRow(contextMenu.curRow, newname)
                    }
                }
                MenuItem {
                    text: "删除条目"
                    onTriggered: {
                        deleteItem()
                    }
                }
            }

            ScrollView {
                anchors.fill: parent
                ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                clip: true
                ColumnLayout {
                    id: treeLayout
                    anchors.fill: parent
                    spacing: 0
                    // width: parent.width // 确保宽度与 ScrollView 一致
                    Repeater {
                        id: tabrepeater
                        model: dialog.tabelM
                        delegate: Loader {
                            // Layout.fillWidth: true
                            Layout.preferredWidth: parent.width // 确保填满父容器宽度

                            sourceComponent: tabComp
                            onLoaded: {
                                // item.tabname = tabname
                                // item.groupmodel = groups
                                // item.tabindex = index
                            }
                            Binding {
                                target: item
                                property: "tabindex"
                                value: index
                            }
                            Binding {
                                target: item
                                property: "groupmodel"
                                value: groups
                            }
                            Binding {
                                target: item
                                property: "tabname"
                                value: tabname
                            }
                        }
                    }
                }
                
            }

            Component {
                id: triangleArrow
                Canvas {
                    id: triangleCanvas
                    width: 20
                    height: 20
                    signal clicked() // 自定义信号
                    property bool isExpanded: true // 初始状态为展开
                    onPaint: {
                        var ctx = getContext("2d");
                        ctx.clearRect(0, 0, width, height); // 清除画布
                        
                        // 设置旋转中心为画布的中心
                        ctx.save();
                        ctx.translate(width / 2, height / 2); // 移动到中心点
                        ctx.rotate(rotationAngle);  // 旋转三角形
                        
                        // 绘制三角形（初始指向右侧）
                        ctx.beginPath();
                        ctx.moveTo(2, 0);  // 右顶点
                        ctx.lineTo(-4, -6); // 左上角
                        ctx.lineTo(-4, 6);  // 左下角
                        ctx.closePath();
                        
                        ctx.fillStyle = "gray"; // 设置填充颜色
                        ctx.fill(); // 填充三角形
                        ctx.restore();  // 恢复状态
                    }

                    property real rotationAngle: Math.PI / 2 // 初始角度为90

                    MouseArea {
                        anchors.fill: parent
                        acceptedButtons: Qt.LeftButton
                        onClicked: {
                            // 切换展开/收起状态
                            triangleCanvas.isExpanded = !triangleCanvas.isExpanded;
                            // 根据展开状态来旋转
                            if (triangleCanvas.isExpanded) {
                                triangleCanvas.rotationAngle += Math.PI / 2; // 收起时逆时针旋转90度
                            } else {
                                triangleCanvas.rotationAngle -= Math.PI / 2; // 展开时顺时针旋转90度
                            }
                            // groupComp.visible = triangleCanvas.isExpanded; // 控制子项可见性
                            triangleCanvas.requestPaint(); // 请求重绘
                            triangleCanvas.clicked();
                        }
                    }
                }
            }
            // treeNode：用于显示每个节点及其子节点
            Component {
                id: tabComp

                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    property string tabname: ""
                    property var tabindex: undefined
                    property var groupmodel: undefined

                    Rectangle {
                        id: tabRec
                        //width: parent.width
                        height: 15
                        color: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1 ? "lightblue" : "transparent"
                        Layout.fillWidth: true
                        RowLayout  {
                            spacing: 10
                            anchors.fill: parent
                            
                            Loader {
                                id: tabtriangleArrowLoader
                                height: parent.height
                                Layout.preferredWidth: 20
                                // Layout.fillWidth: true
                                // anchors.verticalCenter: parent.verticalCenter
                                sourceComponent: triangleArrow
                            }
                            Connections {
                                target: tabtriangleArrowLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                function onClicked() {
                                    groupComp.visible = tabtriangleArrowLoader.item.isExpanded
                                }
                            }
                            Rectangle {
                                height: parent.height
                                Layout.fillWidth: true
                                color: "transparent"

                                Loader {
                                    id: tabtextEditLoader
                                    anchors.fill: parent
                                    sourceComponent: textEditComp
                                    Binding {
                                        target: tabtextEditLoader.item
                                        property: "textContent"
                                        value: tabname
                                    }
                                    Binding {
                                        target: tabtextEditLoader.item
                                        property: "textClr"
                                        value: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === -1 && rootRec.currentParamRow === -1 ? "black" : "lightgray"
                                    }
                                }
                                Connections {
                                    target: tabtextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                    function onEditfinish() {
                                        dialog.tabelM.setData(dialog.tabelM.index(tabindex, 0), tabtextEditLoader.item.textContent, Model.ROLE_PARAM_NAME)
                                    }
                                }
                                // Text {
                                //     text: tabname
                                // }

                                MouseArea {
                                    anchors.fill: parent
                                    enabled: !tabtextEditLoader.item.editMode
                                    acceptedButtons: Qt.AllButtons
                                    propagateComposedEvents: true
                                    onClicked: {
                                        rootRec.currentTabRow = tabindex
                                        rootRec.currentGroupRow = -1
                                        rootRec.currentParamRow = -1
                                        // parent.forceActiveFocus()
                                        midRect1.forceActiveFocus()
                                    }
                                    // 右键弹出菜单
                                    onPressed: if (mouse.button === Qt.RightButton) {
                                        // contextMenu.curRow = tabindex
                                        // contextMenu.modelRef = dialog.customuimodel.tabModel()
                                        // console.log(dialog.customuimodel.tabModel().data(contextMenu.curIdx, Model.ROLE_PARAM_NAME))
                                        contextMenu.x = mouse.x + parent.mapToItem(treeviewItem, 0, 0).x; // 使用全局坐标
                                        contextMenu.y = mouse.y + parent.mapToItem(treeviewItem, 0, 0).y;
                                        contextMenu.open();
                                    }
                                    onDoubleClicked: {
                                        tabtextEditLoader.item.editMode = true;
                                        tabtextEditLoader.item.inputfield.forceActiveFocus();
                                    }
                                }
                            }
                        }
                    }

                    // 子节点递归显示
                    ColumnLayout {
                        id: groupComp
                        spacing: 0
                        Layout.fillWidth: true
                        Layout.leftMargin: 20 // 设置缩进
                        visible: true

                        Repeater {
                            model: groupmodel
                            delegate:  

                            ColumnLayout {
                                spacing: 0
                                Layout.fillWidth: true
                                property int groupindex: index
                                // Layout.leftMargin: 20 // 设置缩进

                                Rectangle {
                                    id: groupRec
                                    //width: parent.width
                                    height: 15
                                    color: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === groupindex && rootRec.currentParamRow === -1 ? "lightblue" : "transparent"
                                    Layout.fillWidth: true
                                    // Layout.leftMargin: 20 // 设置缩进

                                    RowLayout {
                                        spacing: 10
                                        anchors.fill: parent

                                        Loader {
                                            id: grouptriangleArrowLoader
                                            height: parent.height
                                            // Layout.fillWidth: true
                                            Layout.preferredWidth: 20
                                            // anchors.verticalCenter: parent.verticalCenter
                                            sourceComponent: triangleArrow
                                        }
                                        Connections {
                                            target: grouptriangleArrowLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                            function onClicked() {
                                                paramComp.visible = grouptriangleArrowLoader.item.isExpanded
                                            }
                                        }
                                        Rectangle {
                                            height: parent.height
                                            Layout.fillWidth: true
                                            color: "transparent"

                                            Loader {
                                                id: grouptextEditLoader
                                                anchors.fill: parent
                                                sourceComponent: textEditComp
                                                Binding {
                                                    target: grouptextEditLoader.item
                                                    property: "textContent"
                                                    value: groupname
                                                }
                                                Binding {
                                                    target: grouptextEditLoader.item
                                                    property: "textClr"
                                                    value: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === groupindex && rootRec.currentParamRow === -1 ? "black" : "lightgray"
                                                }
                                            }
                                            Connections {
                                                target: grouptextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                                function onEditfinish() {
                                                    groupmodel.setData(groupmodel.index(groupindex, 0), grouptextEditLoader.item.textContent, Qt.DisplayRole)
                                                }
                                            }
                                            // Text {
                                            //     text: groupname
                                            // }

                                            MouseArea {
                                                anchors.fill: parent
                                                enabled: !grouptextEditLoader.item.editMode
                                                acceptedButtons: Qt.AllButtons
                                                propagateComposedEvents: true
                                                onClicked: {
                                                    rootRec.currentTabRow = tabindex
                                                    rootRec.currentGroupRow = groupindex
                                                    rootRec.currentParamRow = -1
                                                    // parent.forceActiveFocus()
                                                    midRect1.forceActiveFocus()
                                                }
                                                // 右键弹出菜单
                                                onPressed: if (mouse.button === Qt.RightButton) {
                                                    // contextMenu.curRow = groupindex
                                                    // contextMenu.modelRef = groupmodel
                                                    contextMenu.x = mouse.x + parent.mapToItem(treeviewItem, 0, 0).x; // 使用全局坐标
                                                    contextMenu.y = mouse.y + parent.mapToItem(treeviewItem, 0, 0).y;
                                                    contextMenu.open();
                                                }
                                                onDoubleClicked: {
                                                    grouptextEditLoader.item.editMode = true;
                                                    grouptextEditLoader.item.inputfield.forceActiveFocus();
                                                }
                                            }
                                        }
                                    }
                                }
                                ColumnLayout {
                                    id: paramComp
                                    spacing: 0
                                    Layout.fillWidth: true
                                    Layout.leftMargin: 20 // 设置缩进
                                    property string nodeName: ""
                                    property int indent: 20

                                    Repeater {
                                        id: params_repeater
                                        model: params
                                        delegate: Rectangle {
                                            id:paramRec
                                            //width: parent.width
                                            height: 15
                                            color: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === groupindex && rootRec.currentParamRow === index ? "lightblue" : "transparent"
                                            Layout.fillWidth: true
                                            // Layout.leftMargin: 40 // 设置缩进

                                            RowLayout {
                                                spacing: 10
                                                anchors.fill: parent
                                                Image {
                                                    source: model.paramIcon
                                                    height: parent.height
                                                    Layout.preferredWidth: 15
                                                    Layout.leftMargin: 5
                                                    // width: parent.height
                                                    // height: parent.height
                                                    // anchors.verticalCenter: parent.verticalCenter
                                                }
                                                Loader {
                                                    id: paramtextEditLoader
                                                    height: parent.height
                                                    Layout.fillWidth: true
                                                    Layout.leftMargin: 5
                                                    // anchors.fill: parent
                                                    // anchors.leftMargin: 20
                                                    sourceComponent: textEditComp
                                                    Binding {
                                                        target: paramtextEditLoader.item
                                                        property: "textContent"
                                                        value: name
                                                    }
                                                    Binding {
                                                        target: paramtextEditLoader.item
                                                        property: "textClr"
                                                        value: rootRec.currentTabRow === tabindex & rootRec.currentGroupRow === groupindex && rootRec.currentParamRow === index ? "black" : "lightgray"
                                                    }
                                                }
                                                Connections {
                                                    target: paramtextEditLoader.item  // 这里使用 checkboxLoader.item 来引用加载后的复选框组件
                                                    function onEditfinish() {
                                                        params.setData(params.index(index, 0), paramtextEditLoader.item.textContent, Model.ROLE_PARAM_NAME)
                                                    }
                                                }
                                                // Text {
                                                //     //id: toggle
                                                //     anchors.verticalCenter : parent.verticalCenter
                                                //     text : paramname
                                                // }
                                            }

                                            MouseArea {
                                                anchors.fill: parent
                                                enabled: !paramtextEditLoader.item.editMode
                                                acceptedButtons: Qt.AllButtons
                                                propagateComposedEvents: true
                                                onClicked: {
                                                    // // 单击选中条目
                                                    // for (let i = 0; i < childrenNodes.count; i++) {
                                                    //     childrenNodes.setProperty(i, "selected", false);
                                                    // }
                                                    // childrenNodes.setProperty(index, "selected", true);
                                                    rootRec.currentTabRow = tabindex
                                                    rootRec.currentGroupRow = groupindex
                                                    rootRec.currentParamRow = index
                                                    // parent.forceActiveFocus()
                                                    midRect1.forceActiveFocus()
                                                }

                                                // 右键弹出菜单
                                                onPressed: if (mouse.button === Qt.RightButton) {
                                                    // contextMenu.curRow = index
                                                    // contextMenu.modelRef = params
                                                    contextMenu.x = mouse.x + paramRec.mapToItem(treeviewItem, 0, 0).x; // 使用全局坐标
                                                    contextMenu.y = mouse.y + paramRec.mapToItem(treeviewItem, 0, 0).y;
                                                    contextMenu.open();
                                                }
                                                onDoubleClicked: {
                                                    paramtextEditLoader.item.editMode = true;
                                                    paramtextEditLoader.item.inputfield.forceActiveFocus();
                                                }
                                            }


                                        }
                                    }
                                }

                            }
                        }
                    }
                }
            }
        }
    }
}