import QtQuick 2.15

Rectangle {
    id: thisScene
    anchors.fill: parent
    color: '#222'

    property var selectedChildren: []
    property ZenoHalfLink halfLink: null

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

    function createLink(args) {
        args.scene = thisScene
        compZenoLink.createObject(thisScene, args)
    }

    function linkInput(input) {
        if (halfLink == null) {
            halfLink = compZenoHalfLink.createObject(thisScene, {
                srcSocket: null,
                dstSocket: input,
                mousePos: input.getPos(),
            })
        } else {
            if (halfLink.srcSocket != null) {
                createLink({
                    srcSocket: halfLink.srcSocket,
                    dstSocket: input,
                })
            }
            linkDestroy()
        }
    }

    function linkDestroy() {
        if (halfLink != null) {
            halfLink.destroy()
            halfLink = null
        }
    }

    function linkOutput(output) {
        if (halfLink == null) {
            halfLink = compZenoHalfLink.createObject(thisScene, {
                srcSocket: output,
                dstSocket: null,
                mousePos: output.getPos(),
            })
        } else {
            if (halfLink.dstSocket != null) {
                createLink({
                    srcSocket: output,
                    dstSocket: halfLink.dstSocket,
                })
            }
            linkDestroy()
        }
    }

    function mousePosition(mpos) {
        if (halfLink != null) {
            halfLink.mousePos = mpos
        }
    }

    Component {
        id: compZenoNode
        ZenoNode {}
    }

    Component {
        id: compZenoLink
        ZenoLink {}
    }

    Component {
        id: compZenoHalfLink
        ZenoHalfLink {}
    }

    Flickable {
        anchors.fill: parent
        boundsBehavior: Flickable.StopAtBounds
        clip: true
        interactive: true

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true

            onClicked: {
                if (halfLink != null) {
                    thisScene.linkDestroy()
                } else {
                    thisScene.doSelect(null)
                }
            }

            onPositionChanged: {
                thisScene.mousePosition(Qt.point(mouseX, mouseY))
            }
        }
    }

    Component.onCompleted: {
        thisScene.addNode({
            title: 'readobj',
            x: 64,
            y: 128,
        })
        thisScene.addNode({
            title: 'transform',
            x: 352,
            y: 64,
        })
    }
}
