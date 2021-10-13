import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.12

Rectangle {
    id: zenoNode
    width: 400
    height: 120
    radius: 5.0
    color: '#555'
    border.color: 'orange'
    border.width: selected ? 4 : 0

    property bool selected: false

    MouseArea {
        anchors.fill: parent
        drag.target: parent
        onClicked: {
            zenoScene.doSelect(zenoNode)
        }
    }
}
