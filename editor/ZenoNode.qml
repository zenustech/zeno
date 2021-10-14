import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: thisNode
    width: 250
    height: 120
    radius: 5.0
    color: '#555'
    border.color: 'orange'
    border.width: selected ? 4 : 0

    property var scene: null
    property alias title: label.text
    property bool selected: false

    Label {
        id: label
        anchors.topMargin: 3
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        text: '(untitled)'
        color: '#ccc'
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 22
    }

    MouseArea {
        anchors.fill: parent
        drag.target: parent
        onClicked: {
            scene.doSelect(thisNode)
        }
    }
}
