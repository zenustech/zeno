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
    property var inputs: []
    property var outputs: []

    function addInput(args) {
        args.node = thisNode
        args.ypos = 48 + 30 * inputs.length
        inputs.push(compZenoInput.createObject(thisNode, args))
        thisNode.height = 38 + 30 * Math.max(inputs.length, outputs.length)
    }

    function addOutput(args) {
        args.node = thisNode
        args.ypos = 48 + 30 * outputs.length
        outputs.push(compZenoOutput.createObject(thisNode, args))
        thisNode.height = 38 + 30 * Math.max(inputs.length, outputs.length)
    }

    function detachNode() {
        for (var i in inputs)
            inputs[i].clearLinks()
        for (var i in outputs)
            outputs[i].clearLinks()
    }

    function deleteThisNode() {
        scene.removeNode(thisNode)
    }

    Component {
        id: compZenoInput
        ZenoInput {}
    }

    Component {
        id: compZenoOutput
        ZenoOutput {}
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
            anchors.topMargin: 2
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
            scene.doSelect(thisNode, mouse.modifiers & Qt.ShiftModifier)
        }
    }

    Component.onCompleted: {
        thisNode.addInput({
            title: 'path',
        })
        thisNode.addInput({
            title: 'options',
        })
        thisNode.addOutput({
            title: 'mesh',
        })
    }
}
