import QtQuick 2.12
import QtQuick.Controls 2.0

Rectangle {
    id: container
    implicitWidth: inputField.width
    implicitHeight: inputField.height
    property alias text: inputField.text
    color: "transparent"
    border.color: inputField.activeFocus ? "#4B9EF4" : "gray"  // 焦点时改变边框颜色
    border.width: inputField.activeFocus ? 2 : 1

    signal editingFinished

    TextInput {
        id: inputField
        anchors.fill: parent

        verticalAlignment: Text.AlignVCenter

        width: 48
        height: 24

        topPadding: 5
        bottomPadding: 5
        leftPadding: 5
        rightPadding: 5
        font.pixelSize: 14
        color: "white"
        focus: true // 确保输入框可以获得焦点
        selectByMouse: true // 启用鼠标选择文本功能
        selectionColor: "#4B9EF4"  // 选中的背景色
        clip: true

        onTextChanged: {
                    
        }

        Keys.onReturnPressed: {
                // 按下回车键时失去焦点
            focus = false;
        }

        onEditingFinished: {
            container.editingFinished()
        }
    }
}