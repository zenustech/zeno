import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: thisSocket
    anchors.topMargin: ypos
    anchors.top: parent.top
    anchors.left: parent.left

    property var node: null
    property alias title: label.text
    property var ypos: 0

    Rectangle {
        id: port
        anchors.right: parent.left
        y: -8
        color: '#aaa'
        width: 20
        height: 20

        MouseArea {
            anchors.fill: parent
            onClicked: {
                scene.linkInput(thisSocket)
            }
        }
    }

    Label {
        id: label
        anchors.leftMargin: 6
        anchors.rightMargin: 6
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        text: '(untitled input)'
        color: '#ccc'
        horizontalAlignment: Text.AlignHLeft
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 22
    }
}
