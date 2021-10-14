import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: thisSocket
    anchors.topMargin: ypos
    anchors.top: parent.top
    anchors.left: parent.left
    anchors.right: parent.right

    property var node: null
    property alias title: label.text
    property var ypos: 0
    property bool selected: false

    Label {
        id: label
        anchors.leftMargin: 6
        anchors.rightMargin: 6
        anchors.fill: parent
        text: '(untitled socket)'
        color: '#ccc'
        horizontalAlignment: Text.AlignHLeft
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 22
    }
}
