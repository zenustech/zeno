import QtQuick 2.15

Rectangle {
    id: thisScene
    anchors.fill: parent
    color: '#222'

    property var selectedChildren: []
    property var pendingLinkedSocket: null

    function doSelect(item, multiselect) {
        if (item == null) {
            for (var i in selectedChildren)
                selectedChildren[i].selected = false
            selectedChildren = []
        } else if (multiselect) {
            if (!selectedChildren.includes(item)) {
                selectedChildren.push(item)
                item.selected = true
            } else {
                selectedChildren.remove(item)
                item.selected = false
            }
        } else {
            if (selectedChildren.length == 1 && selectedChildren.includes(item)) {
                for (var i in selectedChildren)
                    selectedChildren[i].selected = false
                selectedChildren = []
                item.selected = false
            } else {
                for (var i in selectedChildren)
                    selectedChildren[i].selected = false
                selectedChildren = [item]
                item.selected = true
            }
        }
    }

    function addNode(args) {
        args.scene = thisScene
        compZenoNode.createObject(thisScene, args)
    }

    function linkInput(input) {
        compZenoLink.createObject(thisScene, {
            srcSocket: null,
            dstSocket: input,
        })
    }

    function linkOutput(output) {
        compZenoLink.createObject(thisScene, {
            srcSocket: output,
            dstSocket: null,
        })
    }

    Component {
        id: compZenoNode
        ZenoNode {}
    }

    Component {
        id: compZenoLink
        ZenoLink {}
    }

    Flickable {
        anchors.fill: parent
        boundsBehavior: Flickable.StopAtBounds
        clip: true
        interactive: true

        MouseArea {
            anchors.fill: parent

            onClicked: {
                thisScene.doSelect(null)
            }
        }
    }

    Component.onCompleted: {
        thisScene.addNode({
            title: 'readobj',
            x: 64,
            y: 32,
        })
    }
}
