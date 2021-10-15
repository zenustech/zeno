import QtQuick 2.15

ZenoEdge {
    id: thisHalfLink

    property var scene: null
    property ZenoOutput srcSocket: null
    property ZenoInput dstSocket: null
    property point mousePos: Qt.point(0, 0)

    src: srcSocket == null ? mousePos : srcSocket.getPos()
    dst: dstSocket == null ? mousePos : dstSocket.getPos()
}
