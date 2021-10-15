import QtQuick 2.15

Item {
    id: thisEdge

    property point src: Qt.point(300, 300)
    property point dst: Qt.point(400, 200)
    property color lineColor: '#6cf'
    property real lineWidth: 5

    Rectangle {
        id: line
        antialiasing: true
        color: thisEdge.lineColor

        property alias src: thisEdge.src
        property alias dst: thisEdge.dst

        width: thisEdge.lineWidth
        height: Math.hypot(dst.x - src.x, dst.y - src.y)
        rotation: Math.atan2(dst.x - src.x, src.y - dst.y) * (180 / Math.PI)
        x: dst.x - height / 2 * Math.sin(rotation * (Math.PI / 180)) - width / 2
        y: dst.y - height / 2 * (1 - Math.cos(rotation * (Math.PI / 180)))
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
