import QtQuick                   2.12
import QtQuick.Controls          2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Layouts           1.3

import QuickQanava 2.0 as Qan
import "qrc:/QuickQanava" as Qan
import zeno.enum 1.0

Column {
    id: comp
    property bool isEditing: false
    property string text: ""
    property bool handle_mouseevent: true
    property var fontsize: 16

    Text {
        id: displayText
        visible: !comp.isEditing
        text: comp.text
        color: "#FFFFFF"
        font.pixelSize: comp.fontsize

        MouseArea{
            anchors.fill: parent
            // onDoubleClicked: comp.isEditing = true
            onPressed: {
                if (!comp.handle_mouseevent) {
                    mouse.accepted = false
                    return
                }
                else{
                    comp.isEditing = true
                }
            }
        }
    }

    Rectangle {
        visible: comp.isEditing
        color: "lightgray"
        width: editableText.width
        height: editableText.height
        border.color: editableText.activeFocus ? "blue" : "gray"  // 焦点时改变边框颜色
        border.width: 1

        TextInput {
            id: editableText
            anchors.centerIn: parent
            //anchors.fill: parent
            anchors.leftMargin: 8
            anchors.rightMargin: 8
            verticalAlignment: Text.AlignVCenter
            text: comp.text
            focus: comp.isEditing // 自动聚焦
            // width: 120
            height: 24

            // 监听 focus 属性的变化
            onFocusChanged: {
                if (!focus) {
                    //console.log("focus out, comp.text: " + comp.text)
                    comp.isEditing = false; // 退出编辑模式
                    if (comp.text != text) {
                        comp.text = text
                    }
                }
            }

            Keys.onReturnPressed: {
                comp.isEditing = false // 回车退出编辑模式
                if (comp.text != text) {
                    comp.text = text
                }
            }
        }
    }
}