import QtQuick 2.15

Rectangle {
    id: thisNode
    width: 250
    height: 120
    radius: 5.0
    color: '#555'
    border.color: 'orange'
    border.width: selected ? 4 : 0

    property var scene: null
    property bool selected: false

    MouseArea {
        anchors.fill: parent
        drag.target: parent
        onClicked: {
            scene.doSelect(thisNode)
        }
    }
}
