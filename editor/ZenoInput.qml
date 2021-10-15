import QtQuick 2.15
import QtQuick.Controls 2.15

Item {
    id: thisSocket
    anchors.topMargin: ypos
    anchors.top: parent.top
    anchors.left: parent.left

    property var node: null
    property alias title: label.text
    property real ypos: 0
    property var links: []

    function getPos() {
        return Qt.point(parent.x - 10, parent.y + this.ypos)
    }

    function attachLink(link) {
        for (var i in links)
            links[i].deleteThisLink()
        links = []
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
        anchors.right: parent.left
        y: -8
        color: '#aaa'
        width: 20
        height: 20
        radius: 5

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
