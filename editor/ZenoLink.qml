import QtQuick 2.15

ZenoEdge {
    id: thisLink
    color: selected ? 'orange' : '#6cf'
    z: 1

    property var scene: null
    property ZenoOutput srcSocket: null
    property ZenoInput dstSocket: null
    property bool selected: false

    src: srcSocket.getPos()
    dst: dstSocket.getPos()

    function deleteThisLink() {
        scene.removeLink(thisLink)
    }

    function dumpLink() {
        var r_link = {}
        r_link.src = srcSocket.title
        r_link.dst = dstSocket.title
        return r_link
    }

    MouseArea {
        anchors.fill: parent
        onClicked: {
            scene.doSelect(thisLink, mouse.modifiers & Qt.ShiftModifier)
        }
    }
}

/*
dx = x + h/2 * sin(r)
dy = y + h/2 * (1 - cos(r))
sx = x + h/2 * (2 - sin(r))
sy = y - h/2 * (1 + cos(r))

h = sqrt((dx-sx)**2 + (dy-sy)**2)
r = Math.atan2(dx - sx, dy - sy)


dx - h/2 * sin(r) = sx - h/2 * (2 - sin(r))
dx - sx + h = h * sin(r)
sin(r) = (dx - sx)/h + 1


dy + h/2 * (cos(r) - 1) = sy + h/2 * (cos(r) + 1)
dy - sy + h/2 * cos(r) - h/2 = sy + h/2 * cos(r) + h/2
dy - sy + h/2 - sy - h/2 = h * cos(r)
cos(r) = (dy - 2 * sy)/h
*/
