import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: thisSocket
    anchors.topMargin: ypos
    anchors.top: parent.top
    anchors.right: parent.right

    property var node: null
    property alias name: label.text
    property real ypos: 0
    property var links: []

    function getPos() {
        return Qt.point(parent.x + parent.width + port.width / 2, parent.y + ypos)
    }

    function attachLink(link) {
        links.push(link)
    }

    function removeLink(link) {
        links = links.filter(function (e) { return e != link })
    }

    function clearLinks() {
        for (var i in links)
            links[i].deleteThisLink()
        links = []
    }

    Rectangle {
        id: port
        anchors.left: parent.right
        anchors.verticalCenter: parent.verticalCenter
        color: '#aaa'
        width: 20
        height: 20
        radius: 5

        MouseArea {
            anchors.fill: parent
            onClicked: {
                scene.linkOutput(thisSocket)
            }
        }
    }

    Label {
        id: label
        anchors.leftMargin: 6
        anchors.rightMargin: 6
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        text: '(untitled output)'
        color: '#ccc'
        horizontalAlignment: Text.AlignHLeft
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: 22
    }
}
