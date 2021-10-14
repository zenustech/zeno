import QtQuick 2.15
import QtQuick.Controls 2.15

Rectangle {
    id: thisNode
    width: 200
    height: 30
    radius: 5.0
    color: '#444'

    property var scene: null
    property alias title: label.text
    property bool selected: false
    property var inputSockets: []

    function addInputSocket(args) {
        args.node = thisNode
        args.ypos = 48 + 30 * inputSockets.length
        inputSockets.push(compZenoInputSocket.createObject(thisNode, args))
        thisNode.height = 38 + 30 * inputSockets.length
    }

    Component {
        id: compZenoInputSocket
        ZenoSocket {}
    }

    Rectangle {
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 32
        radius: thisNode.radius
        color: '#555'

        Rectangle {
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            height: parent.radius
            color: parent.color
        }

        Label {
            id: label
            anchors.topMargin: 3
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            text: '(untitled node)'
            color: '#ccc'
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            font.pixelSize: 22
        }
    }

    Rectangle {
        anchors.fill: parent
        border.color: 'orange'
        border.width: 4
        radius: parent.radius
        color: 'transparent'
        visible: parent.selected
    }

    MouseArea {
        anchors.fill: parent
        drag.target: parent
        onClicked: {
            scene.doSelect(thisNode)
        }
    }

    Component.onCompleted: {
        thisNode.addInputSocket({
            title: 'path',
        })
        thisNode.addInputSocket({
            title: 'options',
        })
    }
}
