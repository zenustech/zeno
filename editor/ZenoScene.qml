import QtQuick 2.15

Rectangle {
    id: thisScene
    color: '#222'

    property point viewOrigin: Qt.point(-100, 0)
    property real viewScale: 1.2

    property var selectedChildren: []
    property ZenoHalfLink halfLink: null
    property var nodes: []
    property var links: []

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
                selectedChildren = selectedChildren.filter(function(e) { return e != item })
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

    function dumpScene() {
        var r_nodes = []
        for (var i in nodes) {
            var r_node = nodes[i].dumpNode()
            r_nodes.push(r_node)
        }
        var r_scene = {}
        r_scene.view = {}
        r_scene.view.x = viewOrigin.x
        r_scene.view.y = viewOrigin.y
        r_scene.view.scale = viewScale
        r_scene.nodes = r_nodes
        return r_scene
    }

    function removeNode(node) {
        nodes = nodes.filter(function (e) { return e != node })
        node.detachNode()
        node.destroy()
    }

    function removeLink(link) {
        links = links.filter(function (e) { return e != link })
        link.srcSocket.removeLink(link)
        link.dstSocket.removeLink(link)
        link.destroy()
    }

    function deleteSelection() {
        for (var i in selectedChildren) {
            var o = selectedChildren[i]
            if (typeof o.deleteThisLink == 'function')
                o.deleteThisLink()
        }
        for (var i in selectedChildren) {
            var o = selectedChildren[i]
            if (typeof o.deleteThisNode == 'function')
                o.deleteThisNode()
        }
    }

    function addNode(args) {
        args.scene = thisScene
        var node = compZenoNode.createObject(sceneRect, args)
        nodes.push(node)
    }

    function addLink(args) {
        args.scene = thisScene
        var link = compZenoLink.createObject(sceneRect, args)
        args.srcSocket.attachLink(link)
        args.dstSocket.attachLink(link)
        links.push(link)
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
                addLink({
                    srcSocket: halfLink.srcSocket,
                    dstSocket: input,
                })
            }
            linkDestroy()
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
                addLink({
                    srcSocket: output,
                    dstSocket: halfLink.dstSocket,
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

    Item {
        id: sceneRect
        x: -viewOrigin.x
        y: -viewOrigin.y
        scale: viewScale

        MouseArea {
            width: 100000
            height: 100000
            x: -width / 2
            y: -height / 2

            hoverEnabled: halfLink != null

            onClicked: {
                if (halfLink != null) {
                    thisScene.linkDestroy()
                } else {
                    thisScene.doSelect(null)
                }
            }

            onPositionChanged: {
                if (!thisScene.focus)
                    thisScene.focus = true
                var mpos = Qt.point(
                    (mouse.x + x) ,
                    (mouse.y + y) )
                thisScene.mousePosition(mpos)
            }
        }
    }

    Keys.onDeletePressed: {
        deleteSelection()
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
