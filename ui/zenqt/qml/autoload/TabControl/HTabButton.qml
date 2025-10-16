// ==================================================
// =============== 水平标签栏的标签按钮 ===============
// ==================================================

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15


TabButton {

    // 设定值
    property string title: "Unknown TabBtn" // 显示的标题
    property int index: -1 // 在标签栏中的序号
    property bool isLock: app.tab.barIsLock  // 是否已锁定

    // 默认值
    implicitWidth: 150
    implicitHeight: parent.height
    z: checked? 100 : 0 // 选中模式下弹到顶层
    // 信号
    signal toDel(int index) // 关闭按钮的信号
    signal toChecked(int index) // 选中按钮的信号

    signal dragStart(int index) // 开始拖拽的信号
    signal dragFinish(int index) // 结束拖拽的信号
    signal dragMoving(int index, int x) // 拖拽移动的信号

    // 监听选中
    onCheckedChanged: {
        if(checked) toChecked(index) // 发送信号
    }
    Component.onCompleted: {
        if(checked) Qt.callLater(toChecked,index) // 初始选中也发送信号
    }
    

    // 按钮前景
    contentItem: RowLayout {

        // TODO: 图标
        Text{
            Layout.alignment: Qt.AlignLeft
            height: parent.parent.height
            text: ""
        }
        // 标题
        Text {
            Layout.alignment: Qt.AlignCenter // 位于中心
            Layout.fillWidth: true // 填充宽度
            height: parent.parent.height // 适应整个按钮的高度
            text: title // 外部传入的title
            elide: Text.ElideRight // 隐藏超出宽度
            font.bold: parent.checked ? true : false
        }
        // 关闭按钮
        Button {
            // 未锁定，且主按钮悬停或选中时才显示
            visible: !parent.parent.isLock & (parent.parent.hovered | parent.parent.checked)
            Layout.alignment: Qt.AlignRight
            implicitWidth: 24
            implicitHeight: 24
            font.pixelSize: 18
            text: "×"
            background: Rectangle {
                radius: 6
                anchors.fill: parent
                property color bgColorNormal: "#00000000" // 正常
                property color bgColorHovered: "#22000000" // 悬停
                property color bgColorPressed : "#44000000" // 按下
                color: parent.pressed ? bgColorPressed: (
                    parent.hovered ? bgColorHovered : bgColorNormal
                )
            }
            onClicked: {
                toDel(index) // 发送信号
            }
        }
    }

    // 按钮背景
    background: Rectangle {
        anchors.fill: parent
        property color bgColorNormal: "#00000000" // 正常
        property color bgColorHovered: "#11000000" // 悬停
        property color bgColorChecked: "#FFF" // 选中
        color: parent.checked ? bgColorChecked: (
            parent.hovered ? bgColorHovered : bgColorNormal
        )

        MouseArea {
            anchors.fill: parent
            acceptedButtons: Qt.LeftButton | Qt.MiddleButton

            // 拖拽
            drag.target: parent.parent.isLock ? null : parent.parent // 动态启用、禁用拖拽
            drag.axis: Drag.XAxis // 只能沿X轴
            drag.threshold: 50 // 起始阈值
            property bool dragActive: drag.active // 动态记录拖拽状态
            property int dragX: parent.parent.x // 动态记录拖拽时整体的位置
            
            onPressed: { // 左键按下，切换焦点
                if(mouse.button === Qt.LeftButton) {
                    if(! parent.parent.checked){
                        parent.parent.checked = true
                    }
                }
            }
            onClicked: { // 中键点击，删除标签
                if(mouse.button === Qt.MiddleButton && !parent.parent.isLock) {
                    toDel(index) // 发送信号
                }
            }
            onDragActiveChanged: {
                if(drag.active) { // 拖拽开始
                    parent.opacity = 0.6
                    parent.parent.y += parent.parent.height / 2
                    dragStart(index)
                } else { // 拖拽结束
                    parent.opacity = 1
                    parent.parent.y -= parent.parent.height / 2
                    dragFinish(index)
                }
            }
            onDragXChanged: {
                if(drag.active) {
                    dragMoving(index, dragX)
                }
            }
        }

        // 侧边小条
        Rectangle{
            visible: !parent.parent.checked
            height: parent.height-20
            width: 1
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            color: "#000"
        }
    }
}